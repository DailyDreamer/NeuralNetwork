function y = softmax(x)
%y = exp(x) / sum(exp(x));
y = exp(x - log(sum(exp(x))));
%y = exp(x - max(x)) / sum(x);