classdef fcLayer
    %CONVLAYER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        n; % output feature number
        m; % input feature number
        
        W; % Weight
        b; % bias
        activate;
        batchSize;
        input;
        output;
        
        % refer to the comments in convLayer
        lr;
        wc;
        mom;
        grad_W;
        grad_b;
        inc_W;
        inc_b;
        delta;
    end
    
    methods
        function layer = fcLayer(n, m, normStd, activate, lr, wc, mom)
            % Constructor function
            layer.n = n;    %output num
            layer.m = m;    %input num
            layer.W = single(random('norm', 0, normStd, n, m));
            layer.b = zeros(n, 1, 'single');
            layer.inc_W = zeros(size(layer.W), 'single');
            layer.inc_b = zeros(size(layer.b), 'single');
            layer.activate = activate;
            layer.lr = lr;
            layer.wc = wc;
            layer.mom = mom;
        end
        
        function layer = forward(layer, input)
            layer.input = input;
            if (size(input, 3) == 1)            %判断前一层是否是fcLayer
                layer.batchSize = size(input, 2);
            else
                layer.batchSize = size(input, 4);
            end
            assert(numel(input)/layer.batchSize == layer.m);
            tempin = reshape(input,layer.m,layer.batchSize);
            for i = 1:layer.batchSize
                layer.output(:,i) = layer.W * tempin(:,i) + layer.b;
            end
            if strcmp(layer.activate, 'relu')   
                layer.output = relu(layer.output);
            elseif strcmp(layer.activate, 'sigmoid')
               layer.output = sigmoid(layer.output);
            end
            layer.output = single(layer.output);
        end
        
        function layer = backward(layer, delta)
            assert(isequal(size(delta), size(layer.output)));
            tempin = reshape(layer.input,layer.m,layer.batchSize);
            tempGw = 0;
            tempGb = 0;
            insize = size(layer.input);
            switch layer.activate %激活函数的导数
                case 'relu'
                    tempdelta = delta .* (layer.output > 0);
                case 'sigmoid'
                    tempdelta = delta .* layer.output .* (1 - layer.output);
                case 'none'
                    tempdelta = delta;
                otherwise
                    error('!');
            end
            for i = 1:layer.batchSize
                if (size(layer.input, 3) == 1)  %前一层是fcLayer
                    layer.delta(:,i) = reshape(layer.W' * tempdelta(:,i),insize(1),1);
                else                            
                    layer.delta(:,:,:,i) = reshape(layer.W' * tempdelta(:,i),insize(1:3));
                end
                tempGw = tempGw + delta(:,i) * tempin(:,i)';
                tempGb = tempGb + delta(:,i);
            end
            layer.delta = single(layer.delta);
            layer.grad_W = tempGw/layer.batchSize;
            layer.grad_b = tempGb/layer.batchSize;
            % update
            layer.inc_W = layer.mom * layer.inc_W - layer.lr * (layer.grad_W + layer.wc * layer.W);
            layer.W = layer.W + layer.inc_W;
            layer.inc_b = layer.mom * layer.inc_b - 2 * layer.lr * layer.grad_b;
            layer.b = layer.b + layer.inc_b;
        end
    end
end