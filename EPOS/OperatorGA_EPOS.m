function Offspring = OperatorGA_EPOS(Parent,lower,upper,Parameter)
% Crossover and mutation operators of genetic algorithm for EPOS.

    if nargin > 3
        [proC,disC,proM,disM] = deal(Parameter{:});
    else
        [proC,disC,proM,disM] = deal(1,20,1,20);
    end
    if isa(Parent(1),'SOLUTION')
        evaluated = true;
        Parent    = Parent.decs;
    else
        evaluated = false;
    end
    Parent1   = Parent(1:floor(end/2),:);
    Parent2   = Parent(floor(end/2)+1:floor(end/2)*2,:);
    Offspring = zeros(2*size(Parent1,1),size(Parent1,2));
    % Real and integer variables
    Offspring(:,:) = GAreal(Parent1(:,:),Parent2(:,:),lower(:,:),upper(:,:),proC,disC,proM*2*size(Parent1,1)/size(Parent1,2),disM);
    if evaluated
        Offspring = Problem.Evaluation(Offspring);
    end
end

function Offspring = GAreal(Parent1,Parent2,lower,upper,proC,disC,proM,disM)
% Genetic operators for real and integer variables

    %% Simulated binary crossover
    [N,D] = size(Parent1);
    beta  = zeros(N,D);
    mu    = rand(N,D);
    beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
    beta(mu>0.5)  = (2-2*mu(mu>0.5)).^(-1/(disC+1));
    beta = beta.*(-1).^randi([0,1],N,D);
    beta(rand(N,D)<0.5) = 1;
    beta(repmat(rand(N,1)>proC,1,D)) = 1;
    Offspring = [(Parent1+Parent2)/2+beta.*(Parent1-Parent2)/2
                 (Parent1+Parent2)/2-beta.*(Parent1-Parent2)/2];
             
    %% Polynomial mutation
    Lower = repmat(lower,2*N,1);
    Upper = repmat(upper,2*N,1);
    Site  = rand(2*N,D) < proM/D;
    mu    = rand(2*N,D);
    temp  = Site & mu<=0.5;
    Offspring       = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                      (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5; 
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                      (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
end

function Offspring = GAlabel(Parent1,Parent2,proC,proM)
% Genetic operators for label variables

    %% Uniform crossover
    [N,D] = size(Parent1);
    k     = rand(N,D) < 0.5;
    k(repmat(rand(N,1)>proC,1,D)) = false;
    Offspring1    = Parent1;
    Offspring2    = Parent2;
    Offspring1(k) = Parent2(k);
    Offspring2(k) = Parent1(k);
    Offspring     = [Offspring1;Offspring2];
    
    %% Bitwise mutation
    Site = rand(2*N,D) < proM/D;
    Rand = randi(D,2*N,D);
    Offspring(Site) = Rand(Site);
    
    %% Repair
    for i = 1 : 2*N
        Off = zeros(1,D);
        while ~all(Off)
            x = find(~Off,1);
            Off(Offspring(i,:)==Offspring(i,x)) = max(Off) + 1;
        end
        Offspring(i,:) = Off;
    end
end

function Offspring = GAbinary(Parent1,Parent2,proC,proM)
% Genetic operators for binary variables

    %% Uniform crossover
    [N,D] = size(Parent1);
    k     = rand(N,D) < 0.5;
    k(repmat(rand(N,1)>proC,1,D)) = false;
    Offspring1    = Parent1;
    Offspring2    = Parent2;
    Offspring1(k) = Parent2(k);
    Offspring2(k) = Parent1(k);
    Offspring     = [Offspring1;Offspring2];
    
    %% Bit-flip mutation
    Site = rand(2*N,D) < proM/D;
    Offspring(Site) = ~Offspring(Site);
end

function Offspring = GApermutation(Parent1,Parent2,proC)
% Genetic operators for permutation variables

    %% Order crossover
    [N,D]     = size(Parent1);
    Offspring = [Parent1;Parent2];
    k = randi(D,1,2*N);
    for i = 1 : N
        if rand < proC
            Offspring(i,k(i)+1:end)   = setdiff(Parent2(i,:),Parent1(i,1:k(i)),'stable');
            Offspring(i+N,k(i)+1:end) = setdiff(Parent1(i,:),Parent2(i,1:k(i)),'stable');
        end
    end
    
    %% Slight mutation
    k = randi(D,1,2*N);
    s = randi(D,1,2*N);
    for i = 1 : 2*N
        if s(i) < k(i)
            Offspring(i,:) = Offspring(i,[1:s(i)-1,k(i),s(i):k(i)-1,k(i)+1:end]);
        elseif s(i) > k(i)
            Offspring(i,:) = Offspring(i,[1:k(i)-1,k(i)+1:s(i)-1,k(i),s(i):end]);
        end
    end
end