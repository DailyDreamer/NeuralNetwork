function net = init(net)
%init w and b
    inNum = 3;  
    mapsize = [32,32];
    for i = 1 : length(net) 
        net{i}.x = cell(1,inNum);
        switch net{i}.type
            case 'pooling'
                mapsize = mapsize / net{i}.scale;
                assert(all(floor(mapsize)==mapsize), ['Layer ' num2str(i) ' size must be integer instead of ' num2str(mapsize)]);
            case 'conv'
                net{i}.w = cell(inNum, net{i}.numOut);
                net{i}.b = cell(1, net{i}.numOut);
                mapsize = mapsize - net{i}.kernelsize + 1;
                fout = net{i}.numOut * net{i}.kernelsize^2;
                for k = 1 : net{i}.numOut
                    fin = inNum * net{i}.kernelsize^2;
                    for j = 1 : inNum
                        net{i}.w{j,k} = (rand(net{i}.kernelsize) - 0.5) * 2 * sqrt(6 / (fin + fout));
                    end
                    net{i}.b{k} = 0;
                end
                inNum = net{i}.numOut;
            case 'full'
                fin = prod(mapsize) * inNum;    
                fout = net{i}.numOut;
                net{i}.w = (rand(fin, fout) - 0.5) * 2 * sqrt(6 / (fout + fin));
                net{i}.b =  zeros(1, fout);
                fin = fout;
            case 'output'
                fout = 10;      %label num
                net{i}.w = (rand(fin, fout) - 0.5) * 2 * sqrt(6 / (fout + fin));
                net{i}.b =  zeros(1, fout);
            otherwise
                error('!');
        end  
    end  
end  