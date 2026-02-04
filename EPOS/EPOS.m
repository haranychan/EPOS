classdef EPOS < ALGORITHM
% <2025> <single> <real/integer> <expensive/none>
% An Evolutionary Algorithm Assisted by An Ensemble of Pareto-Optimal Surrogate Models
% N             --- 100   --- Population size of DE
% F             --- 0.5   --- Scaling factor of DE
% CR            --- 0.9   --- Crossover rate of DE
% mut           --- 3     --- Mutation strategy ID for DE
% xov           --- 1     --- Crossover strategy ID for DE
% delta         --- 0.2   --- Rate of validation data
% data_times    --- 5     --- Parameter to define data size
% MOEA_N        --- 10    --- Population size of MOEA
% MOEA_omega    --- 10    --- Maximum generations of MOEA
% LCB_a         --- 2     --- Coefficient in LCB

%------------------------------- Reference --------------------------------
% K. Nishihara, Y. Jin, and M. Nakata, "An Evolutionary Algorithm Assisted 
% by An Ensemble of Pareto-Optimal Surrogate Models," IEEE Trans. Cybern.,
% under review, 2025.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [N,F,CR,mut,xov,delta,data_times,MOEA_N,MOEA_omega,LCB_a] ...
                = Algorithm.ParameterSet(100,0.5,0.9,3,1,0.2,5,10,10,2);
            
            D = Problem.D;
            
            %% Generate random population
            PopDec         = UniformPoint(N, D, 'Latin');
            Population     = Problem.Evaluation(repmat(Problem.upper - Problem.lower, N, 1) .* PopDec + repmat(Problem.lower, N, 1), [repmat(Algorithm.metric.runtime, N, 1)]);
            Arc            = Population;

            % Archive
            db_x = Population.decs; db_y = Population.objs;
            
            gen = 1;
            
            %% Optimization
            while Algorithm.NotTerminated(Arc)
                
                % DE
                [B, I] = sort(db_y); 
                pop = db_x(I(1:N), :); fit = B(1:N);
                cand = DEGetTrialVector_EPOS(pop, fit, Problem.lower, Problem.upper, F, CR, mut, xov);
                
                % Get data
                train_x = db_x(I(1:min(data_times*D, size(db_x, 1))), :); train_y = B(1:min(data_times*D, size(db_x, 1)));
                c = cvpartition(size(train_x, 1), 'HoldOut', delta);
                idx = c.test; train_Xs = train_x(~idx, :); test_Xs = train_x(idx, :); train_Fs = train_y(~idx, :); test_Fs = train_y(idx, :);

                % MOEA setting
                train_size = size(train_Xs, 1); 
                pair = pdist2(train_Xs, train_Xs);
                MOEA_lower = [train_size/2, min(nonzeros(pair))]; MOEA_upper = [train_size, max(nonzeros(pair))];
                
                % MOEA initalization
                PopDec     = UniformPoint(MOEA_N,2,'Latin');
                PopDec     = repmat(MOEA_upper-MOEA_lower,MOEA_N,1).*PopDec+repmat(MOEA_lower,MOEA_N,1);
                surrs      = cell(MOEA_N, 1); acc = zeros(MOEA_N, 1); cmp = zeros(MOEA_N, 1);
                for n = 1:MOEA_N
                    surrs{n} = OnceNewrb_EPOS(train_Xs.', train_Fs.', 0, PopDec(n,2), round(PopDec(n,1)), Problem.maxFE);
                    pred_Fs  = sim(surrs{n}, test_Xs.').';
                    acc(n)   = sqrt(mean((test_Fs - pred_Fs).^2));
                    cmp(n)   = surrs{n}.layers{1}.size;
                end
                Population = SOLUTION(PopDec, [acc, cmp], zeros(MOEA_N,1), surrs);
                [~,FrontNo,CrowdDis] = NSGAIIEnvironmentalSelection_EPOS(Population,MOEA_N);
                
                for o = 1:MOEA_omega
                    MatingPool = TournamentSelection(2,MOEA_N,FrontNo,-CrowdDis);
                    OffDec     = OperatorGA_EPOS(Population(MatingPool).decs,MOEA_lower,MOEA_upper);
                    surrs      = cell(MOEA_N, 1); acc = zeros(MOEA_N, 1); cmp = zeros(MOEA_N, 1);
                    for n = 1 : size(OffDec, 1)
                        surrs{n} = OnceNewrb_EPOS(train_Xs.', train_Fs.', 0, OffDec(n,2), round(OffDec(n,1)), Problem.maxFE);
                        pred_Fs  = sim(surrs{n}, test_Xs.').';
                        acc(n)   = sqrt(mean((test_Fs - pred_Fs).^2));
                        cmp(n)   = surrs{n}.layers{1}.size;
                    end
                    Offspring = SOLUTION(OffDec, [acc, cmp], zeros(size(OffDec, 1), 1), surrs);
                    [Population,FrontNo,CrowdDis] = NSGAIIEnvironmentalSelection_EPOS([Population,Offspring],MOEA_N);
                end

                rank1Idx = find(FrontNo == 1);
                rank1Dec = Population(rank1Idx).decs;
                [~, ia, ~] = unique(rank1Dec, 'stable', 'rows');
                rank1Idx = rank1Idx(ia);
                rank1Srg = Population(rank1Idx).adds;
                                
                % Screen
                
                if size(rank1Idx, 2) == 1
                    surr = rank1Srg;
                    cand_Fs = sim(surr{1}, cand.').';
                    [~, idx] = min(cand_Fs);
                else       
                    cand_each_Fs = zeros(size(cand, 1), size(rank1Idx, 2));
                    for n = 1:size(rank1Idx, 2)
                        surr = rank1Srg(n);
                        cand_each_Fs(:, n) = sim(surr{1}, cand.').';
                    end
                    cand_Fs = mean(cand_each_Fs, 2);
                    
                    cand_diff = sum((cand_each_Fs - cand_Fs).^2, 2);
                    cand_std  = sqrt(cand_diff./(size(rank1Idx, 2)-1));
                    LCB = cand_Fs - LCB_a .* cand_std;
                    [~, idx] = min(LCB);
                end

                for i = 1:size(idx, 1)
                    % Evaluate solutions
                    offspringDec = cand(idx(i), :);
                    offspring = Problem.Evaluation(offspringDec, [Algorithm.metric.runtime]);
                    Arc       = [Arc, offspring];
                    Algorithm.NotTerminated(Arc);
    
                    db_x = [db_x; offspring.dec];  db_y = [db_y; offspring.obj];
                    [db_x, ia, ~] = unique(db_x, 'stable', 'rows'); db_y = db_y(ia);
                end
                
                gen = gen + 1;
            end
        end
    end
end