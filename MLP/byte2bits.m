function bits = byte2bits( byte )
%BYTE2BITS converts a byte to 8 bits in descending order,
% for example, (16)_10 = (00010000)_2

% 预分配空间
bits = zeros(1, 8);
for i = 1:8
    % 通过与、移位操作读取每一位的内容
    bits(9-i) = bitshift(bitand(byte, bitshift(1, i-1)), -(i-1));
end

end

