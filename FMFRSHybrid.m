function [Reduct, SumHyp] = FMFRSHybrid(DT, Theta, Gamma, c_inpu)
%%c_inpu: number of categorical attributes in DT
[dtx,dty] = size(DT);
f_inpu = dty - 1;
n_cls = length(unique(DT(:,dty)));
%disp(n_cls);
inputD = f_inpu - c_inpu;
Num_type = 1:inputD;
Cat_type = (inputD+1):f_inpu;

v_c(1:(fix(dtx/2))) = 0;
a_s(1:n_cls) = 1;
n_eclass = cell(n_cls,1);
n_eclass(1:n_cls,1) = {v_c};

for z = 1:dtx
    ir = DT(z,dty);
    %disp(ir);
    n_eclass{ir}(a_s(ir)) = z;
    a_s(ir) = a_s(ir) + 1;
end
n_eclass = cellfun(@(x) x(x>0),n_eclass,'UniformOutput',false);

M = zeros(2,inputD);
HBS = cell(1,n_cls);         % Hyperboxes in each classes
A = cell(1,300);
A(1:300) = {M};
HBS(1:n_cls) = {A};

HbObjIndex = cell(1,n_cls); % Index of objects belonging to Hyperbox
B = cell(1,300);
p(1:(fix(dtx/4))) = 0;
B(1:300) = {p};
HbObjIndex(1:n_cls) = {B};
HypEachClass(1:n_cls) = 0;   % Number of hyperbox in each class

%% calling hypCreation function for each class
for iter = 1:n_cls
    HypCreation(n_eclass{iter},iter);
    HBS{iter} = HBS{iter}(1:HypEachClass(iter));
    HbObjIndex{iter} = HbObjIndex{iter}(1:HypEachClass(iter));
    HbObjIndex{iter} =  cellfun(@(x) x(x>0),HbObjIndex{iter},'UniformOutput',false);
end
clear n_eclass;  %clear memories

%% Overlap Testing of Hyperboxes and Creation of FDM
t = 0;
lr = n_cls - 1;
for b = 1:lr
    t = t + HypEachClass(b) * sum(HypEachClass((b+1):n_cls));
end

cat_ele = cell(1,f_inpu);
cat_info = zeros(1,f_inpu);
for att = Cat_type
    cat_ele{att} = unique(DT(:,att))';
    cat_info(att) = length(cat_ele{att});
end

rs = 1;
DM = zeros(t,f_inpu);
for N = 1:(n_cls-1)
    for W = (N+1):n_cls
        for E = 1:(HypEachClass(N))
            for Y = 1:(HypEachClass(W))
                DM(rs,Num_type) = IntervalEachDimOverlapTest(HBS{N}{E},HBS{W}{Y}, inputD);   %make sure tt is vector
                h1 = HbObjIndex{N}{E};
                h2 = HbObjIndex{W}{Y};
                l1 = length(h1);
                l2 = length(h2);
                if c_inpu ~= 0
                    for b = Cat_type     % Categorical DM Construction
                        uq1 = unique(DT(h1,b))';
                        if length(uq1) == cat_info(b)
                            [freq1, ~] = histcounts(DT(h1,b), [uq1, max(uq1)+1]);  % count frequency
                        else
                            freq1 = zeros(1,cat_info(b));
                            [fre, ~] = histcounts(DT(h1,b), [uq1, max(uq1)+1]);  % count frequency
                            for ele = 1:length(cat_ele{b})
                                if ismember(cat_ele{b}(ele), uq1)
                                    freq1(ele) = fre(find(uq1 == cat_ele{b}(ele)));
                                end
                            end
                        end
                        v1 = freq1/l1;    % each object ration w.r.t. total number of object in hyperbox
                        
                        % for hyperbox h2
                        uq2 = unique(DT(h2,b))';
                        if length(uq2) == cat_info(b)
                            [freq2, ~] = histcounts(DT(h2,b), [uq2, max(uq2)+1]);
                        else
                            freq2 = zeros(1,cat_info(b));
                            [fre, ~] = histcounts(DT(h2,b), [uq2, max(uq2)+1]);  % count frequency
                            for ele = 1:length(cat_ele{b})
                                if ismember(cat_ele{b}(ele), uq2)
                                    freq2(ele) = fre(find(uq2 == cat_ele{b}(ele)));
                                end
                            end
                        end
                        %[freq2, ~] = histcounts(DT(h2,b), [uq2, max(uq2)+1]);
                        v2 = freq2/l2;
                        
                        DM(rs,b)= 1-(dot(v1, v2) / (norm(v1) * norm(v2)));    %Cosine similarity
                    end
                end
                rs = rs + 1;
            end
        end
    end
end
clear HBS;  % clear memories
clear HbObjIndex;


%% Sequential Forward selection approach SAT to compute a reduct
Red = [];
Inp = 1:f_inpu;
l = sum(DM);
[~,ll] = max(l);
Red = [Red ll];

snv = arrayfun(@Smin,sum(DM,2));
tt = (DM(:,ll) == snv);
DM = DM(~tt,:);
snv(tt) = [];
snv = snv';

while(~isempty(DM))
    Mx = 0;
    MxI = [];
    Rt = setdiff(Inp,Red);
    index = 0;
    for K = Rt
        nv = arrayfun(@Smin,sum(DM(:,[Red K]),2));
        nvv = sum(nv);
        if (nvv > Mx)
            MxI = nv;
            Mx = nvv;
            index = K;
        end
    end
    MxI = MxI';
    Red = [Red index];
    DM = DM(MxI~= snv,:);
    snv(MxI == snv) = [];
end

%% Final Reduct
Reduct = Red;
SumHyp = sum(HypEachClass);

%% function for hyperbox creation in a particular class
    function HypCreation(dataR,Class)
        NumOfHyp = 1;
        ExceedHypNums = 1;
        HbNumObj(1:300) = 0;   % Number of element in each hyperbox
        for Input = dataR
            InputData = DT(Input,(1:inputD));
            isMember = false;
            if((NumOfHyp) >= (300*ExceedHypNums))
                HBS{Class}((end+1):(end+300)) = {M};
                HbObjIndex{Class}(end:(end+300)) = {p};
                ExceedHypNums = ExceedHypNums + 1;
                HbNumObj((end+1):(end+300)) = 0;
            end
            if (NumOfHyp == 1)
                HBS{Class}{NumOfHyp}(1,:) = InputData;
                HBS{Class}{NumOfHyp}(2,:) = InputData;
                HbNumObj(NumOfHyp) = HbNumObj(NumOfHyp) + 1;
                HbObjIndex{Class}{NumOfHyp}(HbNumObj(NumOfHyp)) = Input;
                HypEachClass(Class) = 1;
                NumOfHyp = NumOfHyp +1;
            else
                HypMembValues(1:HypEachClass(Class)) = 0;
                for HbIndex = 1:HypEachClass(Class)
                    result = (sum(max(0,(1-max(0,Gamma*min(1,(HBS{Class}{HbIndex}(1,:) - InputData)))))) + sum(max(0,(1-max(0,Gamma*min(1,InputData-(HBS{Class}{HbIndex}(2,:))))))));
                    result = result/(2*inputD);
                    if(result == 1)
                        isMember = true;
                        HbNumObj(HbIndex) = HbNumObj(HbIndex) + 1;
                        HbObjIndex{Class}{HbIndex}(HbNumObj(HbIndex)) = Input;
                        break;
                    else
                        HypMembValues(HbIndex) = result;
                    end
                end
                if(isMember ~= true)
                    [~,r_m] = sort(HypMembValues,'descend');
                    for k = 1:HypEachClass(Class)
                        hboxExpanded = r_m(k);
                        MaxV = max(HBS{Class}{hboxExpanded}(2,:),InputData);
                        MinV = min(HBS{Class}{hboxExpanded}(1,:),InputData);
                        threshold = sum(MaxV - MinV);
                        if((inputD * Theta) >= threshold)
                            HBS{Class}{hboxExpanded}(2,:) = MaxV;
                            HBS{Class}{hboxExpanded}(1,:) = MinV;
                            HbNumObj(hboxExpanded) = HbNumObj(hboxExpanded) + 1;
                            HbObjIndex{Class}{hboxExpanded}(HbNumObj(hboxExpanded)) = Input;
                            isMember = true;
                            break;
                        end
                    end
                end
                if (~isMember)
                    HBS{Class}{NumOfHyp}(1,:) = InputData;
                    HBS{Class}{NumOfHyp}(2,:) = InputData;
                    HypEachClass(Class) = HypEachClass(Class) + 1;
                    HbNumObj(NumOfHyp) = HbNumObj(NumOfHyp) + 1;
                    HbObjIndex{Class}{NumOfHyp}(HbNumObj(NumOfHyp)) = Input;
                    NumOfHyp = NumOfHyp + 1;
                end
            end
        end
    end
end
