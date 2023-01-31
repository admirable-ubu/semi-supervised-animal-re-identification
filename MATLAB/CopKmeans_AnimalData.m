clear, clc, close all

path1 = '..\features\features\';

Videos = {'Pigs_49651_960_540_500f','Koi_5652_952_540',...
    'Pigeons_8234_1280_720','Pigeons_4927_960_540_600f',...
    'Pigeons_29033_960_540_300f'};

Videos = sort(Videos);

flag_basu2 = false;

FeatureSets = {'AE','HOG','LBP','MN2','RGB'};

nT = 5; % times to run with the same number of constraints

rng(1959)

number_of_constraints = [10, 50, 100, 300, 700];

for numcon = 1:numel(number_of_constraints)

    Acc = zeros(5,5,2); % accuracies for seeded and constrained
    % <Reshape to account of all feature sets

    nML = number_of_constraints(numcon);
    nCL = nML; % number of constraints

    ColNames = {};


    for folds = 1:2

        for i = 1:5 % Videos

            fprintf('Video %s  fold %i Constrints %i\n',...
                Videos{i},folds,nML*2)

            % Load up constraint files
            fnc1 = [path1,'h',num2str(3-folds),'_constraints_',...
                Videos{i},'.csv']; % file with constraints
            ConstraintTable1 = readtable(fnc1);
            fnc2 = [path1,'h',num2str(3-folds),'_BB_',...
                Videos{i},'.csv']; % file with constraints
            ConstraintTable2 = readtable(fnc2);

            % Constraint data
            ConsData.Overlap = table2array(ConstraintTable1);
            % first BB, second BB, overlap
            ConsData.Overlap(:,1:2) = ConsData.Overlap(:,1:2) + 1;
            % + 1 is needed because the index comes

            ConsData.Frames = table2array(ConstraintTable2(:,5));

            for j = 1:5 % Features
                partfn = [Videos{i}, '_', FeatureSets{j}];
                fn = [path1,'h',num2str(folds),'_', partfn,'.csv'];
                if folds == 1
                    ColNames = [ColNames, partfn];
                end
                T = readtable(fn);
                trd = table2array(T(:,1:end-2)); % Seeds
                trl = T.Labels;
                fn = [path1,'h',num2str(3-folds),'_', Videos{i}, ...
                    '_', FeatureSets{j},'.csv'];
                T = readtable(fn);
                tsd = table2array(T(:,1:end-2));
                tsl = T.Labels;
                labels_all = [trl;tsl];

                % cop-kmeans ------------

                acc = [];

                k = 1;

                while k <= nT % runs with the same number of constraints

                    % Pick constraints
                    [ML,CL] = pick_constraints(ConsData,nML,nCL);

                    if flag_basu2
                        % BASU 2
                        seed_index = 1:numel(trl);
                        test_labels = cop_kmeans_basu2([trd;tsd], ...
                            seed_index, labels_all(seed_index), ...
                            ML, CL, 100);
                        test_labels = test_labels(numel(seed_index)+1:end);
                    else
                        % COP KMEANS
                        me = grpstats(trd,trl,"mean"); %#ok<*UNRCH>
                        test_labels_raw = cop_kmeans(tsd, ML, CL, 100, me);

                        unique_labels = unique(trl);
                        test_labels = unique_labels(test_labels_raw);
                    end

                    % Guard against no return from cop-kmeans
                    if ~isempty(test_labels)
                        acc(k) = mean(tsl == ...
                            test_labels); %#ok<*SAGROW>
                        k = k + 1;
                    end
                end

                Acc(i,j,folds)  = mean(acc);
            end
        end
    end

    fprintf('\nExperiments with %i constraints completed.\n\n',2*nCL)
    temp = mean(Acc,3);
    AverageAccuracyCOPKmeans(numcon,:) = temp(:)';
end

% Save in csv file --------------------------------------------------------
ToSave = AverageAccuracyCOPKmeans;
Table = array2table(ToSave,'VariableNames',ColNames,'RowNames',...
    string(number_of_constraints));

if flag_basu2
    writetable(Table,...
        'ResultsConstrainedClustering_copkmeans_basu2.csv',...
        'WriteRowNames',true)
else
    writetable(Table,...
        'ResultsConstrainedClustering_seeded_copkmeans.csv',...
        'WriteRowNames',true)
end