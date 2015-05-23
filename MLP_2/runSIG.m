LayerCell = {3072, 800, 500, 10};
learnRate = 0.1;
momentum = 0.1;
epochNum = 1;
LayerNum = length(LayerCell);
x = cell(LayerNum,1);

%training model
[w,b] = trainingSIG(LayerCell, learnRate, momentum, epochNum);

importfile('test_batch.mat');  %导入变量
data_test = double(data) / 255;
indexs = zeros(1,size(data_test, 1));
right = 0;
for n = 1:size(data_test, 1)
    x{1} = data_test(n,:);
    %forward
    for i = 1:LayerNum-1
        x{i+1} = sigmoid (x{i} * w{i}+b{i});
    end;
    y = x{LayerNum};
    %final output
    indexs(n) = find(y == max(y));  %输出
    if (indexs(n) == labels(n)+1)
        right = right + 1;
    end
end;
right_rate = right / size(data_test, 1);
display(right_rate);