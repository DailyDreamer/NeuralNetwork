function [w,b] = trainingSIG(LayerCell, learnRate, momentum, epochNum)
LayerNum = length(LayerCell); 
importfile('OURDIG.PNT.mat');  %导入变量
images = evalin('base','images');
chars = evalin('base','chars');
% intial w, b
w = cell(LayerNum-1,1);
dw = cell(LayerNum-1,1);
b = cell(LayerNum-1,1);
db = cell(LayerNum-1,1);
x = cell(LayerNum,1);
delta = cell(1,LayerNum-1);
lastError = zeros([1,LayerCell{LayerNum}]);
for i = 1:LayerNum-1
    numIn = LayerCell{i};
    numOut = LayerCell{i+1};
    boundry = sqrt(6 / (numIn+numOut));
    w{i} = random('Uniform', -boundry, boundry, numIn, numOut);
    b{i} = random('Uniform', -boundry, boundry, 1,numOut);
    dw{i} = zeros(numIn, numOut);
    db{i} = zeros(1, numOut);
end
for epoch = 1:epochNum
    disp(epoch);
    if epoch > 20
        learnRate = learnRate * 0.1;
        momentum = momentum * 0.1;
    end
    index = randperm(length(images));
    for in = 1:length(images)-1000
        n = index(in);
        x{1} = images{n};
        %forward
        for i = 1:LayerNum-1
            x{i+1} = sigmoid (x{i} * w{i}+b{i});
        end;
        y = x{LayerNum};
        %backward
        T = zeros([1,LayerCell{LayerNum}]);   %教师信号
        T(chars(n)- '0'+1) = 1;
        currentError = 0.5 * sum((T-y).^2);
         %停止准则
%        if ((abs(lastError-currentError) < 0.0005 * abs(lastError)) & count > 6000)
%            done = true;
%            break;
%        end
        lastError = currentError;
        tempDelta = T - y;      %local grad
        for i = LayerNum-1 : -1 : 1
            delta{i} = x{i+1} .* (1-x{i+1}) .* tempDelta;
            tempDelta = delta{i} * w{i}';
            w{i} = w{i} + momentum * dw{i} + learnRate *  x{i}' * delta{i};
            b{i} = b{i} + momentum * db{i} + learnRate * delta{i};
            dw{i} = learnRate *  x{i}' * delta{i};
            db{i} = learnRate * delta{i};
        end
    end
    right = 0;
    for in = length(images)-1000+1:length(images)
        n = index(in);
        x{1} = images{n};
        %forward
        for i = 1:LayerNum-1
            x{i+1} = sigmoid (x{i} * w{i}+b{i});
        end;
        y = x{LayerNum};
        temp = find(y == max(y));  %输出
        if (temp == (chars(n) -'0'+ 1))
            right = right + 1;
        end
    end
    right_rate = right / 1000
save('learningData.mat', 'w', 'b');
end
disp('learning finished');
