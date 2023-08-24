clear; close all; 


% General Parameters 
f_au_t = 1378999.4; %freq of an oscillation of perdiod 1 au in cm^-1
dt = 100; %simulation time-step 
fsf = f_au_t/dt; %frequency scaling factor
fmax = 3500; %upper bound to plot


step = [0:2:14]; %initial : step : final







%% Raw plotting

%figure creation
fig = figure; 
ax = axes(fig);
xlim([0,fmax])
set(ax, 'YTick', zeros(1,0))
title("IR spectrum for H_2O molecules", 'FontSize',18)
set(ax, 'FontSize',16)

hold on
for i=step

    file_loc = sprintf("/Volumes/THEO/MAGISTRALE/Materie/Atomistic Simulation Methods/Exam/dipole/dip%04i.dat",i);
    
    %Find the options of the file
    opts = detectImportOptions(file_loc);
    %modify the Variabe Names 
    opts.VariableNames = {'ex','ey','ez'};
    
    %read the data
    d = readtable(file_loc,opts);

    % computing the power spectrum of each column of d
    [spx,f] = pspectrum(d.ex,fsf); 
    [spy,f] = pspectrum(d.ey,fsf); 
    [spz,f] = pspectrum(d.ez,fsf); 
    
    % compute the IR spectrum 
    s = (f.^2) .* (spx + spy + spz); 
    
    plot(f,s)

end
hold off


%% Filtered plot

%figure creation
fig = figure; 
ax = axes(fig);
xlim([0,fmax])
set(ax, 'YTick', zeros(1,0))
title("IR spectrum for H_2O molecules", 'FontSize',18)
set(ax, 'FontSize',16)
xlabel("frequency [cm^{-1}]")
ylabel("IR absorpion [a.u]")
box on;

hold on
h = create_filter(100);
for i=step

    file_loc = sprintf("/Volumes/THEO/MAGISTRALE/Materie/Atomistic Simulation Methods/Exam/dipole/dip%04i.dat",i);
    
    %Find the options of the file
    opts = detectImportOptions(file_loc);
    %modify the Variabe Names 
    opts.VariableNames = {'ex','ey','ez'};
    
    %read the data
    d = readtable(file_loc,opts);

    % computing the power spectrum of each column of d
    [spx,f] = pspectrum(d.ex,fsf); 
    [spy,f] = pspectrum(d.ey,fsf); 
    [spz,f] = pspectrum(d.ez,fsf); 
    
    % compute the IR spectrum 
    s = (f.^2) .* (spx + spy + spz); 
    s_filtered = conv(h,s,'same');
    
    plot(f,s_filtered,'LineWidth',1)
    xlim([0,fmax])

end
hold off








function h = create_filter(alpha)
    x = linspace(-5,5,4096);
    h = exp(-alpha*x.^2);
end







