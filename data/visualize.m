% data = csvread('raw_data.csv', 1);
% plot(data(2:ceil(length(data)/32)-1,11) + data(2:ceil(length(data)/32)-1,12) -data(1:ceil(length(data)/32)-2,11) - data(1:ceil(length(data)/32)-2,12))
% plot(data(ceil(length(data)/32):ceil(2*length(data)/32)-1,11) + data(ceil(length(data)/32):ceil(2*length(data)/32)-1,12) -data(ceil(length(data)/32)-1:ceil(2*length(data)/32)-2,11) - data(ceil(length(data)/32)-1:ceil(length(data)/32)-2,12))
function visualize(idx, task)
    data = csvread(['seperatePaths/',int2str(idx),'.csv'],1);
    n = size(data, 1);
    if task == 1
        plot(data(2:n,11) + data(2:n,12) - data(1:n-1,11) - data(1:n-1,12))
    elseif task == 2
        plot(data(2:n,24) - data(1:n-1,24))
    elseif task == 3
        plot(data(2:n,72) - data(1:n-1,72))
end