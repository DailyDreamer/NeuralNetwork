function [w,b] = trainingSIG(LayerCell, learnRate, momentum, epochNum)
trainFile = {'data_batch_1.mat','data_batch_2.mat','data_batch_3.mat','data_batch_4.mat','data_batch_5.mat'};
LayerNum = length(LayerCell); 
% intial w, b
w = cell(LayerNum-1,1);
dw = cell(LayerNum-1,1);
b = cell(LayerNum-1,1);
db = cell(LayerNum-1,1);
x = cell(LayerNum,1);
delta = cell(1,LayerNum-1);
%lastError = zeros([1,LayerCell{LayerNum}]);
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
    for bat = 1:length(trainFile)
        load (trainFile{bat});
        data_train = double(data) / 255;
        for n = 1:size(data_train, 1)
            x{1} = data_train(n,:);
            %forward
            for i = 1:LayerNum-1
                x{i+1} = sigmoid (x{i} * w{i}+b{i});
            end;
            y = x{LayerNum};
            %backward
            T = zeros([1,LayerCell{LayerNum}]);   %教师信号
            T(labels(n)+1) = 1;
            currentError = 0.5 * sum((T-y).^2)
             %停止准则
        %    if ((abs(lastError-currentError) < 0.0005 * abs(lastError)) & count > 6000)
        %        done = true;
        %        break;
        %    end
        %    lastError = currentError;
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
    end
    %validate
    load (trainFile{5});
    data_test = double(data) / 255;
    right = 0;
    for n = 1:size(data_test, 1)
        x{1} = data_test(n,:);
        %forward
        for i = 1:LayerNum-1
            x{i+1} = sigmoid (x{i} * w{i}+b{i});
        end;
        y = x{LayerNum};
        %final output
        temp = find(y == max(y));  %输出
        if (temp == labels(n)+1)
            right = right + 1;
        end
    end;
    save('learningData.mat', 'w', 'b');
    right_rate = right / length(labels)
end
disp('learning finished');
