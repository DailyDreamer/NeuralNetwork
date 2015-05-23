classdef softmaxLayer
    %TEMP Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        prob;
        pred;
        loss;
        delta;
        output;
        accuracy;
    end
    
    methods
        function layer = forward(layer, input)
            % Insert your code
            for i = 1:size(input,2)
                layer.output(:,i) = softmax(input(:,i));
            end
        end
        
        function layer = backward(layer, label)
            % Insert your code
            tempLoss = 0;
            tempAcc = 0;
            for i = 1:length(label)
                T = zeros(10,1);
                T(label(i)) = 1;
                tempLoss = tempLoss - sum(T .* log(layer.output(:,i)));
                layer.delta(:,i) = layer.output(:,i) - T;
                templabel = find(layer.output(:,i) == max(layer.output(:,i)));
                if (label(i) == templabel)
                   tempAcc = tempAcc + 1;
                end
            end
            layer.accuracy = tempAcc / length(label);
            layer.loss = tempLoss / length(label);
        end
    end
end

