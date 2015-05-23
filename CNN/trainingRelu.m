function net = trainingRelu(net, learnRate, epochNum)

trainFile = {'data_batch_1.mat','data_batch_2.mat','data_batch_3.mat','data_batch_4.mat','data_batch_5.mat'};
input = cell(1,3);
for epoch = 1:epochNum
    disp(epoch);
    for bat = 1:length(trainFile)-1
        load (trainFile{bat});
        for index = 1:length(labels)
            temp = data(index, :);
            input{1} = double(reshape(temp(1:1024), [32,32])')/255 - 1;
            input{2} = double(reshape(temp(1025:2048), [32,32])')/255 - 1;
            input{3} = double(reshape(temp(2049:3072), [32,32])')/255 - 1;
            %forward
            [net, output, mapsize] = forwardRelu(net, input);
            %backward
            %init T
            T = zeros(1,10);
            T(labels(index)+1) = 1;
            currentError = - sum(T .* log(output));
            outdelta = output - T;
            for i = length(net):-1:1
                switch net{i}.type
                    case 'conv'
                        for j = 1:length(net{i}.x)
                            net{i}.delta{j} = 0;
                            for k = 1:net{i}.numOut
                                net{i}.delta{j} = net{i}.delta{j} + conv2(net{i+1}.delta{k}, net{i}.w{j,k}, 'full') .* (net{i}.x{j} > 0);
                                net{i}.w{j,k} = net{i}.w{j,k} - learnRate * conv2(net{i}.x{j}, rot90(net{i+1}.delta{k},2),'valid');
                            end
                        end
                        for k = 1:net{i}.numOut
                            net{i}.b{k} = net{i}.b{k} - learnRate * net{i+1}.delta{k}; 
                        end
                    case 'pooling'
                        for j = 1:length(net{i}.x)
                            net{i}.delta{j} = (net{i}.x{j} > 0) .* kron(net{i+1}.delta{j}, ones(net{i}.scale)) / net{i}.scale^2;
                        end
                    case 'full'
                        net{i}.tdelta = net{i+1}.delta * net{i}.w' .* (net{i}.tx > 0);   % [1,fin] =  [1,fout] * [fout,fin] .* [1,fin]
                        net{i}.delta = mat2cell(reshape(net{i}.tdelta, mapsize.*[1,length(net{i}.x)]),mapsize(1), ones(1,length(net{i}.x))*mapsize(2)); %reshape tempdelta to input maps
                        net{i}.w = net{i}.w - learnRate *  net{i}.tx' * net{i+1}.delta;     %[fin,fout] = [fin,1] * [1,fout]
                        net{i}.b = net{i}.b - learnRate * net{i+1}.delta;  %[1,fout]
                    case 'output'
                        net{i}.delta = outdelta * net{i}.w' .* (net{i}.x > 0);
                        net{i}.w = net{i}.w - learnRate *  net{i}.x' * outdelta;     %[fin,fout] = [fin,1] * [1,fout]
                        net{i}.b = net{i}.b - learnRate * outdelta;  %[1,fout]
                    otherwise
                        error('!');
                end
            end
        end
    end
    %validate
    load (trainFile{5});
    right = 0;
    for index = 1:length(labels)
        temp = data(index, :);
        input{1} = double(reshape(temp(1:1024), [32,32])')/255 - 1;
        input{2} = double(reshape(temp(1025:2048), [32,32])')/255 - 1;
        input{3} = double(reshape(temp(2049:3072), [32,32])')/255 - 1;
        %forward
        [net, output] = forwardRelu(net, input);
        tempLabel = find(output == max(output)) - 1;  %Êä³ö
        if (labels(index) == tempLabel)
            right = right + 1;
        end
    end
    save('learningData.mat','net');
    right_rate = right / length(labels)
end
disp('learning finished');
