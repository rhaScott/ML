function plotData(x, y)

%   PLOTDATA(x,y) plots the data points and gives the figure axes labels of
%   population and profit.

figure; % open a new figure window

% Plot the data
plot(x, y, 'rx', 'MarkerSize', 10); 

% Set the y-axis label  
ylabel('Profit in $10,000s'); 
  
% Set the x-axis label
xlabel('Population of City in 10,000s'); 

end
