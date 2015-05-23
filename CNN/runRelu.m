%struct of net
net = {                                                     
struct('type', 'conv', 'numOut', 16, 'kernelsize', 5) 
struct('type', 'pooling', 'scale', 2) 
struct('type', 'conv', 'numOut', 16, 'kernelsize', 5) 
struct('type', 'pooling', 'scale', 2)
struct('type', 'conv', 'numOut', 32, 'kernelsize', 4)
struct('type', 'pooling', 'scale', 2)
struct('type', 'full', 'numOut', 1024)
struct('type', 'output')
};
learnRate = 0.001;
epochNum = 9;
%init w and b
net = init(net);
%training
net = trainingRelu(net, learnRate, epochNum);
%test
load test_batch.mat;
input = cell(1,3);
right = 0;
for index = 1:length(labels)
    temp = data(index, :);
    input{1} = double(reshape(temp(1:1024), [32,32])')/255 - 1;
    input{2} = double(reshape(temp(1025:2048), [32,32])')/255 - 1;
    input{3} = double(reshape(temp(2049:3072), [32,32])')/255 - 1;
    [net, output] = forwardRelu(net, input);
    tempLabel = find(output == max(output)) - 1;  %Êä³ö
    if (labels(index) == tempLabel)
        right = right + 1;
    end
end;
right_rate = right / length(labels);
display(right_rate);