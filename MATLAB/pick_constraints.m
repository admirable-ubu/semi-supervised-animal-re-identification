function [ML, CL] = pick_constraints(ConsData, nML, nCL)

Ov = ConsData.Overlap;
Ov = Ov(Ov(:,3) > 0.5,:); % reduce to the required overlap threshold

nML = min(nML,size(Ov,1));


ML_index = randperm(size(Ov,1),nML); % choose ML constraints
ML = Ov(ML_index,1:2); 
% from Python :)

F = ConsData.Frames;
uf = unique(F);
CL = [];
while size(CL,1) < nCL
    % Pick a frame with more than 1 BB
    chosen_frame = randi(numel(uf));
    instances = find(F == uf(chosen_frame));
    while numel(instances) < 2
        chosen_frame = randi(numel(uf));
        instances = find(F == uf(chosen_frame));
    end
    new_CL = instances(sort(randperm(numel(instances),2)))';

    if isempty(CL)
        CL = new_CL;
    elseif ~ismember(new_CL,CL,'rows')
        CL = [CL;new_CL];
    end
end



