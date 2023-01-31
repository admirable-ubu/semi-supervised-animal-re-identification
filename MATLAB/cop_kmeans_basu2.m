function [labels, iters, centres] = cop_kmeans_basu2(data,seed_index,...
    seed_labels, ML, CL, maxIter)
%
% COP_KMEANS: Returns the labels and centres of clusters applying 
% must-link and cannot-link constraints. 
%
% data: a numerical array of size N(objects)-by-n(features)
% ML is an nML-by-2 array containing the indices of the pairs that 
%       must be in the same cluster
% CL: an nCL-by-2 array containing the indices of the pairs 
%       that cannot be in the same cluster.
% maxIter: the limit number of iterations of the k-means algorithm;
%       default = 1000
% initial_means: an array of size number_of_clusters-by-n with initial
%       means. If not specified, random rows of data are used instead 
%
% For the method, see
% [Wagstaff, K., Cardie, C., Rogers, S., & Schrödl, S., Constrained 
% k-means clustering with background knowledge. In ICML, Vol. 1, 2001, 
% pp. 577-584.]
% https://web.cse.msu.edu/~cse802/notes/ConstrainedKmeans.pdf
%

%========================================================================
% (c) L. Kuncheva                                                   ^--^
% 20.10.2022 -----------------------------------------------------  \oo/
% -------------------------------------------------------------------\/-%

% The number of clusters is determined by the seed labels (assuming seed
% labels are integers - no need to be consecutive or to start with 1)
%
% flag_constrained = 1 for *Constrained* kmeans and 0 for *Seeded* kmeans

% Relabel the seeded part to 1,2, ... 
% Revert the labels later.
seed_labels = grp2idx(categorical(seed_labels));

% Initial means
unique_labels = unique(seed_labels);
number_of_clusters = numel(unique_labels);
me = grpstats(data(seed_index,:),seed_labels,"mean");

ML = ML + numel(seed_labels);
CL = CL + numel(seed_labels);

in_constraints = unique([CL;ML]); 
% all objects in constraints 
in_constraints = in_constraints(randperm(numel(in_constraints))); 

old_me = me-1; % old means

iter = 1;

while any(old_me~=me,"all") && iter<maxIter
    iter = iter + 1;
    old_me = me;    
    e = pdist2(data,me);
    [~,labels] = min(e,[],2); % labels 

    labels(seed_index) = seed_labels; % preserve the seed labels

    new_labels = labels;

    for something = 1:numel(in_constraints)
        jj = in_constraints(something);

         di = pdist2(data(jj,:),me);
        [~,isorted] = sort(di);
        
        j = 1; % cluster index
        not_done = true;
        while j <= number_of_clusters && not_done
            if check_validity_point(jj,isorted(j),...
                    new_labels,ML,CL)
                new_labels(jj) = isorted(j);
                not_done = false;
            end
            j = j + 1;
        end
        if ~new_labels(jj)
            disp('Impossible clustering.')
            labels = [];
            centres = [];
            return
        else
            labels(jj) = new_labels(jj);
        end
    end
    
    nm = grpstats(data,labels,"mean"); % new means
    uc = unique(labels); % to avoid empty clusters - keep old means
    me(uc,:) = nm;

end

labels = unique_labels(labels);
centres = me;
iters = iter;

end

% -------------------------------------------------------------------------
function out = check_validity_point(point_index,point_class, labels, ...
    ML,CL)
out = true;

for i = 1:size(ML,1)
    if ML(i,1) == point_index && labels(ML(i,2))~=point_class ...
            && labels(ML(i,2))~= 0
        out = false;
        return
    elseif ML(i,2) == point_index && ...
            labels(ML(i,1))~=point_class && labels(ML(i,1))~= 0
        out = false;
        return
    end
end

for i = 1:size(CL,1)
    if CL(i,1) == point_index && labels(CL(i,2))==point_class ...
            && labels(CL(i,2))~= 0
        out = false;
        return
    elseif CL(i,2) == point_index && ...
            labels(CL(i,1))==point_class && labels(CL(i,1))~= 0
        out = false;
        return
    end
end

end
