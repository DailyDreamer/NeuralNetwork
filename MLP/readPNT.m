function y = readPNT(fileName)
PNTFile = fileName;
fid = fopen(PNTFile, 'r');
images = cell(0);
chars = char(0);
count = 0;
while 1
    len = fread(fid, 1, 'uint16');
    if isempty(len)
        break;
    end
    sharp = fread(fid, 2, 'char=>char');
    if chars(1) == char(0)
        chars (1) = sharp(2);
    else
        chars = [chars; sharp(2)];
    end
    width = fread(fid, 1, 'uint8'); height = fread(fid, 1, 'uint8');
    num = width*height; content = fread(fid, num/8, 'uint8');
    im = zeros(1, num);
    for i = 1:num/8
        im(8*(i-1) + 1: 8*i) = byte2bits(content(i)); 
    end
    im = reshape(im, width, height)';
    
    col = zeros([1,width]);
    row = zeros([height,1]);
    for i = 1:width
        for j = 1:height
            col(1,i) = col(1,i) + im(j,i);
            row(j,1) = row(j,1) + im(j,i);
        end
    end
    total = sum(sum(im));
    tw = 0;
    for i = 1:width
        nw = i;
        if (tw > total*0.95) & (col(1,i) == 0)
            break;
        end
        tw = tw + col(1,i);
    end
    th = 0;
    for i = 1:height
        nh = i;
        if (th > total*0.95) & (row(i,1) == 0)
            break;
        end
        th = th + row(i,1);
    end
    if nw < nh / 4
        nw = nh;
    end
    if nh < nw / 4
        nh = nw;
    end
    sample = 6;
    tempw = round(nw/16);
    temph = round(nh/16);
    newim = zeros(16,16);
    for j = 1:height/sample
        for i = 1:width/sample
            if sum(sum(im((j-1)*temph+1:j*temph,(i-1)*tempw+1:i*tempw))) > tempw * temph * 0.5
                newim(j,i) = 1;
            else
                newim(j,i) = 0;
            end
        end
    end
% % 展示图像内容,可以注释掉
    % imagesc(newim);colormap('gray');
    % pause(0.2);
     
   newim = reshape(newim, 1, num/(sample*sample));
   images = [images; newim]; 
   count = count + 1
end
save([PNTFile '.mat'], 'images', 'chars');

fclose(fid);