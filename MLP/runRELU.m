LayerCell = {16*16, 512, 128, 10};
learnRate = 0.0005;
momentum = 0.01;
epochNum = 30;
LayerNum = length(LayerCell);
x = cell(LayerNum,1);
%training model
[w,b] = trainingRELU(LayerCell, learnRate, momentum, epochNum);

importfile('coltest.pnt.mat');  %导入变量
indexs = zeros(1,length(images));
right = 0;
for n = 1:length(images)
    x{1} = images{n};
    %forward
    for i = 1:LayerNum-2
        x{i+1} = relu (x{i} * w{i}+b{i});
    end;
    y = softmax(x{LayerNum-1} * w{LayerNum-1}+b{LayerNum-1});
    %final output
    indexs(n) = find(y == max(y));  %输出
    if (indexs(n) == (chars(n) -'0'+ 1))
        right = right + 1;
    end
end;
right_rate = right / length(images);
display(right_rate);