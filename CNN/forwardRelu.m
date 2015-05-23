function [net, output, mapsize] = forwardRelu(net, input)
net{1}.x = input;
numIn = length(input);
mapsize = [32,32];  
for i = 1:length(net)
    switch net{i}.type
        case 'conv'
            mapsize = mapsize - net{i}.kernelsize + 1;
            for k = 1:net{i}.numOut
                temp = zeros(mapsize);
                for j = 1:numIn
                    temp = temp + conv2(net{i}.x{j}, rot90(net{i}.w{j,k},2), 'valid');
                end
                net{i+1}.x{k} = relu(temp + net{i}.b{k});
            end
            numIn = net{i}.numOut;
        case 'pooling'  %average pooling
            mapsize = mapsize / net{i}.scale;
            aveMatrix = ones(net{i}.scale)/net{i}.scale^2;
            for j = 1:numIn
                temp = conv2(net{i}.x{j},aveMatrix,'valid');
                net{i+1}.x{j} = relu(temp(1:net{i}.scale:end, 1:net{i}.scale:end));
            end
        case 'full'
            fin = prod(mapsize) * length(net{i}.x);
            net{i}.tx = reshape(cell2mat(net{i}.x),1,fin);
            net{i+1}.x = relu(net{i}.tx * net{i}.w + net{i}.b);
        case 'output'
            output = softmax(net{i}.x * net{i}.w + net{i}.b);
        otherwise
            error('!');
    end
end