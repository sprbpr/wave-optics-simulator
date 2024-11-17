clc; clear; close all

func_rect = @(x, y) 0 + 1.*((abs(x) <= 2) & (abs(y) <= 0.3));% 1.*((abs(x) <= 1) & (abs(y) <= 2));

L = 10; N = 201;
x1 = linspace(-L/2, L/2, N); y1 = linspace(-L/2, L/2, N);
[X1, Y1] = meshgrid(x1, y1);


% circle hole
if(0)
    L = 10; N = 201;
    x1 = linspace(-L/2, L/2, N); y1 = linspace(-L/2, L/2, N);
    [X1, Y1] = meshgrid(x1, y1);

    func_rect = @(x, y) 0 + 1.*((x.^2 + y.^2) <= 2);% 1.*((abs(x) <= 1) & (abs(y) <= 2));

    u1 = func_rect(X1, 0.75*Y1);

    z1 = 55*L;
    lad0 = L/100;
%     C = 5e3; lad0 = z1/C;
    D = 10*L;
    [u2, x2, y2] = screen(u1, x1, y1, lad0, z1, D, 3);
    

    figure
    colormap('gray');

    subplot(1, 2, 1)
    surf(x1, y1, abs(u1), 'edgecolor', 'none')
    view(0, 90); axis square;
    xlim([min(x1), max(x1)]); ylim([min(y1), max(y1)]);
    title('Input', 'interpreter', 'latex')
    xlabel('x', 'interpreter', 'latex'); ylabel('y', 'interpreter', 'latex');

    subplot(1, 2, 2)
    surf(x2, y2, abs(u2), 'edgecolor', 'none')
    view(0, 90); axis equal;
    xlim([min(x2), max(x2)]); ylim([min(y2), max(y2)]);
    title('Output', 'interpreter', 'latex')
    xlabel('x', 'interpreter', 'latex'); ylabel('y', 'interpreter', 'latex');
    
end

% single slit
if(0)
    L = 10; N = 301;
    x1 = linspace(-L/2, L/2, N); y1 = linspace(-L/2, L/2, N);
    [X1, Y1] = meshgrid(x1, y1);

    func_rect = @(x, y) 0 + 1.*((abs(x) <= 2) & (abs(y) <= 0.3));% 1.*((abs(x) <= 1) & (abs(y) <= 2));

    u1 = func_rect(X1, 0.75*Y1);

    z1 = 25*L;
    lad0 = L/100;
%     C = 5e3; lad0 = z1/C;
    D = 20*L;
    [u2, x2, y2] = screen(u1, x1, y1, lad0, z1, D, 3);

    figure
    colormap('gray'); axis equal

    subplot(1, 2, 1)
    surf(x1, y1, abs(u1), 'edgecolor', 'none')
    view(0, 90); axis equal;
    xlim([min(x1), max(x1)]); ylim([min(y1), max(y1)]);
    title('Input', 'interpreter', 'latex')
    xlabel('x', 'interpreter', 'latex'); ylabel('y', 'interpreter', 'latex');


    subplot(1, 2, 2)
    surf(x2, y2, abs(u2), 'edgecolor', 'none')
    view(0, 90); axis equal;
    xlim([min(x2), max(x2)]); ylim([min(y2), max(y2)]);
    title('Output', 'interpreter', 'latex')
    xlabel('x', 'interpreter', 'latex'); ylabel('y', 'interpreter', 'latex');

end

% single slit Gaussian
if(0)
    L = 10; N = 301;
    x1 = linspace(-L/2, L/2, N); y1 = linspace(-L/2, L/2, N);
    [X1, Y1] = meshgrid(x1, y1);

    func_rect = @(x, y) 0 + 1.*((abs(x) <= 2) & (abs(y) <= 0.3)).*exp(-(x.^2 + y.^2)/0.8);
    u1 = func_rect(X1, 0.75*Y1);

    z1 = 25*L;
    lad0 = L/100;
%     C = 5e3; lad0 = z1/C;
    D = 20*L;
    [u2, x2, y2] = screen(u1, x1, y1, lad0, z1, D, 3);

    figure
    colormap('gray'); axis equal

    subplot(1, 2, 1)
    surf(x1, y1, abs(u1), 'edgecolor', 'none')
    view(0, 90); axis equal;
    xlim([min(x1), max(x1)]); ylim([min(y1), max(y1)]);
    title('Input', 'interpreter', 'latex')
    xlabel('x', 'interpreter', 'latex'); ylabel('y', 'interpreter', 'latex');


    subplot(1, 2, 2)
    surf(x2, y2, abs(u2), 'edgecolor', 'none')
    view(0, 90); axis equal;
    xlim([min(x2), max(x2)]); ylim([min(y2), max(y2)]);
    title('Output', 'interpreter', 'latex')
    xlabel('x', 'interpreter', 'latex'); ylabel('y', 'interpreter', 'latex');

end


% Double slit
if(0)
    func_rect = @(x, y) 0 + 1.*((abs(x) <= 2) & (abs(y) <= 0.3));% 1.*((abs(x) <= 1) & (abs(y) <= 2));
    
    L = 10; N = 201;
    x1 = linspace(-L/2, L/2, N); y1 = linspace(-L/2, L/2, N);
    [X1, Y1] = meshgrid(x1, y1);
    
    
    phi = 0;
    u1 = func_rect(X1, 15*Y1 + 2.5) + exp(1i*phi)*func_rect(X1, 15*Y1 - 2.5);
    
    z1 = 14*L;
    lad0 = L/100;
%     C = 7e3; lad0 = z1/C;
    D = 20*L;
    [u2, x2, y2] = screen(u1, x1, y1, lad0, z1, D, 3);


    figure
    colormap('gray'); axis equal

    subplot(1, 2, 1)
    surf(x1, y1, abs(u1), 'edgecolor', 'none')
    view(0, 90); axis equal;
    xlim([min(x1), max(x1)]); ylim([min(y1), max(y1)]);
    title('Input', 'interpreter', 'latex')
    xlabel('x', 'interpreter', 'latex'); ylabel('y', 'interpreter', 'latex');


    subplot(1, 2, 2)
    surf(x2, y2, abs(u2), 'edgecolor', 'none')
    view(0, 90); axis equal;
    xlim([min(x2), max(x2)]); ylim([min(y2), max(y2)]);
    title('Output', 'interpreter', 'latex')
    xlabel('x', 'interpreter', 'latex'); ylabel('y', 'interpreter', 'latex');
    
end

% Farfield objects (not coherence)
if(0)
    L = 10; N = 301;
    x1 = linspace(-L/2, L/2, N); y1 = linspace(-L/2, L/2, N);
    [X1, Y1] = meshgrid(x1, y1);
    u1 = func_rect(X1, Y1 + 0.5) + func_rect(X1, Y1 - 0.5);

    z1 = 17*L;
    lad0 = L/100;
%     C = 2e3; lad0 = z1/C;

    D = 5*L;
    [u2, x2, y2] = screen(u1, x1, y1, lad0, z1, D, 3);


    figure
    colormap('gray'); axis equal

    subplot(1, 2, 1)
    surf(x1, y1, abs(u1), 'edgecolor', 'none')
    view(0, 90); axis equal;

    subplot(1, 2, 2)
    surf(x2, y2, abs(u2), 'edgecolor', 'none')
    view(0, 90); axis equal;
    
end

% ring apperture
if(0)
    
    L = 10; N = 201;
    x1 = linspace(-L/2, L/2, N); y1 = linspace(-L/2, L/2, N);
    [X1, Y1] = meshgrid(x1, y1);
    
    func_rect = @(x, y) 0 + 1.*((abs(x) <= 2) & (abs(y) <= 0.3));% 1.*((abs(x) <= 1) & (abs(y) <= 2));

%     pic = imread('pic04.jpg');
%     [Ynum, Xnum] = size(pic(:, :, 1));
%     Xnum = (Xnum - 1)*(mod(Xnum, 2) == 0) + Xnum*(mod(Xnum, 2) == 1);
%     Ynum = (Ynum - 1)*(mod(Ynum, 2) == 0) + Ynum*(mod(Ynum, 2) == 1);
%     x1 = -(Ynum - 1)/2:(Ynum - 1)/2;
%     y1 = -(Ynum - 1)/2:(Ynum - 1)/2;
%     pic = (pic >= 200);
%     pic = pic(1:Ynum, 1:Ynum, 1);


%     u1 = pic;
    u1 = func_rect(X1, Y1 + 2);
    
    z1 = 10*L;
    lad0 = L/100;
%     C = 1e3; lad0 = z1/C;
    R = 0.5*L;
    
    [u2, x2, y2] = ring_apperture(u1, x1, y1, R, lad0, z1, 1);
    
    z2 = 77*L;
    D = 10*L; 
    [u3, x3, y3] = screen(u2, x2, y2, lad0, z2, D, 2);
    
    figure
    colormap('gray'); axis equal

    subplot(2, 2, 1)
    surf(x1, y1, abs(u1), 'edgecolor', 'none')
    view(0, 90); axis equal;

    subplot(2, 2, 2)
    surf(x2, y2, abs(u2), 'edgecolor', 'none')
    view(0, 90); axis equal;
    
    subplot(2, 2, 3)
    surf(x3, y3, abs(u3), 'edgecolor', 'none')
    view(0, 90); axis equal;
    
end

% single lens
if(0)
    L = 10; N = 201;
    x1 = linspace(-L/2, L/2, N); y1 = linspace(-L/2, L/2, N);
    [X1, Y1] = meshgrid(x1, y1);
    func_rect = @(x, y) 0 + 1.*((abs(x) <= 2) & (abs(y) <= 0.3));% 1.*((abs(x) <= 1) & (abs(y) <= 2));

%     pic = imread('picpic.png');
%     [Ynum, Xnum] = size(pic(:, :, 1));
%     Xnum = (Xnum - 1)*(mod(Xnum, 2) == 0) + Xnum*(mod(Xnum, 2) == 1);
%     Ynum = (Ynum - 1)*(mod(Ynum, 2) == 0) + Ynum*(mod(Ynum, 2) == 1);
%     x1 = -(Ynum - 1)/2:(Ynum - 1)/2;
%     y1 = -(Ynum - 1)/2:(Ynum - 1)/2;
%     % pic = (pic >= 200);
%     pic = pic(1:Ynum, 1:Ynum, 1);
%     N = Xnum;
%     u1 = rot90(rot90(pic));
%     x1 = linspace(-L/2, L/2, N); y1 = linspace(-L/2, L/2, N);
    u1 = func_rect(X1, Y1 + 2);
    
    z1 = 5*L;
    lad0 = L/100;
%     C = 10e3; lad0 = z1/C;
    f = 0.5*z1;
    R = 3*L;
    
    [u2, x2, y2] = ring_lens(u1, x1, y1, R, lad0, f, z1, 2);
    
    z2 = z1*f/(z1 - f);
    [u3, x3, y3] = screen(u2, x2, y2, lad0, z2, L, 3);
    
    figure
    colormap('gray'); axis equal

    subplot(2, 2, 1)
    surf(x1, y1, abs(u1), 'edgecolor', 'none')
    view(0, 90); axis equal;

    subplot(2, 2, 2)
    surf(x2, y2, abs(u2), 'edgecolor', 'none')
    view(0, 90); axis equal;
    
    subplot(2, 2, 3)
    surf(x3, y3, abs(u3), 'edgecolor', 'none')
    view(0, 90); axis equal;
    
    figure
    colormap('gray'); axis equal

    subplot(1, 2, 1)
    surf(x1, y1, abs(u1), 'edgecolor', 'none')
    view(0, 90); axis equal;
    xlim([min(x1), max(x1)]); ylim([min(y1), max(y1)]);
    title('Input', 'interpreter', 'latex')
    xlabel('x', 'interpreter', 'latex'); ylabel('y', 'interpreter', 'latex');


    subplot(1, 2, 2)
    surf(x3, y3, abs(u3), 'edgecolor', 'none')
    view(0, 90); axis equal;
    xlim([min(x3), max(x3)]); ylim([min(y3), max(y3)]);
    title('Output', 'interpreter', 'latex')
    xlabel('x', 'interpreter', 'latex'); ylabel('y', 'interpreter', 'latex');
    
end

% Fourier of the Image
if(0)
    L = 10; N = 201;
    x1 = linspace(-L/2, L/2, N); y1 = linspace(-L/2, L/2, N);
    [X1, Y1] = meshgrid(x1, y1);
    func_rect = @(x, y) 0 + 1.*((abs(x) <= 2) & (abs(y) <= 0.3));% 1.*((abs(x) <= 1) & (abs(y) <= 2));

    pic = imread('picpic.png');
    [Ynum, Xnum] = size(pic(:, :, 1));
    Xnum = (Xnum - 1)*(mod(Xnum, 2) == 0) + Xnum*(mod(Xnum, 2) == 1);
    Ynum = (Ynum - 1)*(mod(Ynum, 2) == 0) + Ynum*(mod(Ynum, 2) == 1);
    x1 = -(Ynum - 1)/2:(Ynum - 1)/2;
    y1 = -(Ynum - 1)/2:(Ynum - 1)/2;
    % pic = (pic >= 200);
    pic = pic(1:Ynum, 1:Ynum, 1);
    N = Xnum;
    u1 = rot90(rot90(pic));
    x1 = linspace(-L/2, L/2, N); y1 = linspace(-L/2, L/2, N);
    
%     u1 = func_rect(X1, Y1 + 2);
    
    z1 = 10*L;
    lad0 = L/100;
%     C = 5e3; lad0 = z1/C;
    f = z1;
    R = 2*L;
    
    [u2, x2, y2] = ring_lens(u1, x1, y1, R, lad0, f, z1, 3);
    
    z2 = f;
    [u3, x3, y3] = screen(u2, x2, y2, lad0, z2, L, 3);
    
    figure
    colormap('gray'); axis equal

    subplot(2, 2, 1)
    surf(x1, y1, abs(u1), 'edgecolor', 'none')
    view(0, 90); axis equal;

    subplot(2, 2, 2)
    surf(x2, y2, abs(u2), 'edgecolor', 'none')
    view(0, 90); axis equal;
    
    subplot(2, 2, 3)
    surf(x3, y3, abs(u3), 'edgecolor', 'none')
    view(0, 90); axis equal;
    
    figure
    colormap('gray'); axis equal

    subplot(1, 2, 1)
    surf(x1, y1, abs(u1), 'edgecolor', 'none')
    view(0, 90); axis equal;
    xlim([min(x1), max(x1)]); ylim([min(y1), max(y1)]);
    title('Input', 'interpreter', 'latex')
    xlabel('x', 'interpreter', 'latex'); ylabel('y', 'interpreter', 'latex');


    subplot(1, 2, 2)
    surf(x3, y3, abs(u3), 'edgecolor', 'none')
    view(0, 90); axis equal;
    xlim([min(x3), max(x3)]); ylim([min(y3), max(y3)]);
    title('Output', 'interpreter', 'latex')
    xlabel('x', 'interpreter', 'latex'); ylabel('y', 'interpreter', 'latex');

end

% Focusing
if(0)
    L = 10; N = 201;
    x1 = linspace(-L/2, L/2, N); y1 = linspace(-L/2, L/2, N);
    [X1, Y1] = meshgrid(x1, y1);
    func_rect = @(x, y) 0 + 1.*((abs(x) <= 2) & (abs(y) <= 0.3));

    % pic = imread('pic04.jpg');
    % [Ynum, Xnum] = size(pic(:, :, 1));
    % Xnum = (Xnum - 1)*(mod(Xnum, 2) == 0) + Xnum*(mod(Xnum, 2) == 1);
    % Ynum = (Ynum - 1)*(mod(Ynum, 2) == 0) + Ynum*(mod(Ynum, 2) == 1);
    % x1 = -(Ynum - 1)/2:(Ynum - 1)/2;
    % y1 = -(Ynum - 1)/2:(Ynum - 1)/2;
    % pic = (pic >= 200);
    % pic = pic(1:Ynum, 1:Ynum, 1);


    % u1 = rot90(rot90(pic);
    u1 = func_rect(X1, Y1/5);
    
    d = 10*L;
    R_circ = 2;
    lad0 = L/1000;
%     C = 5e3; lad0 = d/C;

    [u2, x2, y2] = object_circ(u1, x1, y1, R_circ, lad0, d, 1);
    
    z1 = 20*L;
    f = 0.5*z1;
    R = 2*L;
    
    [u3, x3, y3] = ring_lens(u2, x2, y2, R, lad0, f, z1, 3);

    distance = z1;
    z2 = distance*f/(distance - f);
    [u4, x4, y4] = screen(u3, x3, y3, lad0, z2, L, 3);
    
    figure
    colormap('gray');

    subplot(2, 2, 1)
    surf(x1, y1, abs(u1), 'edgecolor', 'none')
    view(0, 90); axis equal;
    xlim([min(x1), max(x1)]); ylim([min(y1), max(y1)]);
    xlabel('x', 'interpreter', 'latex'); ylabel('y', 'interpreter', 'latex');

    subplot(2, 2, 2)
    surf(x2, y2, abs(u2), 'edgecolor', 'none')
    view(0, 90); axis equal;
    xlim([min(x2), max(x2)]); ylim([min(y2), max(y2)]);
    xlabel('x', 'interpreter', 'latex'); ylabel('y', 'interpreter', 'latex');

    subplot(2, 2, 3)
    surf(x3, y3, abs(u3), 'edgecolor', 'none')
    view(0, 90); axis equal;
    xlim([min(x3), max(x3)]); ylim([min(y3), max(y3)]);
    xlabel('x', 'interpreter', 'latex'); ylabel('y', 'interpreter', 'latex');

    subplot(2, 2, 4)
    surf(x4, y4, abs(u4), 'edgecolor', 'none')
    view(0, 90); axis equal;
    xlim([min(x4), max(x4)]); ylim([min(y4), max(y4)]);
    xlabel('x', 'interpreter', 'latex'); ylabel('y', 'interpreter', 'latex');
    
    figure
    colormap('gray'); axis equal

    surf(x4, y4, abs(u4), 'edgecolor', 'none')
    view(0, 90); axis equal;
    xlim([min(x4), max(x4)]); ylim([min(y4), max(y4)]);
    title('Output', 'interpreter', 'latex')
    xlabel('x', 'interpreter', 'latex'); ylabel('y', 'interpreter', 'latex');
    
end

% Zone Plate
if(0)
    
end

% two points merge
if(0)
    L = 10; N = 201;
    x1 = linspace(-L/2, L/2, N); y1 = linspace(-L/2, L/2, N);
    [X1, Y1] = meshgrid(x1, y1);
    func_rect = @(x, y) 0 + 1.*((abs(x) <= 2) & (abs(y) <= 0.3));% 1.*((abs(x) <= 1) & (abs(y) <= 2));

    % pic = imread('picpic.png');
    % [Ynum, Xnum] = size(pic(:, :, 1));
    % Xnum = (Xnum - 1)*(mod(Xnum, 2) == 0) + Xnum*(mod(Xnum, 2) == 1);
    % Ynum = (Ynum - 1)*(mod(Ynum, 2) == 0) + Ynum*(mod(Ynum, 2) == 1);
    % x1 = -(Ynum - 1)/2:(Ynum - 1)/2;
    % y1 = -(Ynum - 1)/2:(Ynum - 1)/2;
    % % pic = (pic >= 200);
    % pic = pic(1:Ynum, 1:Ynum, 1);

    % u1 = pic;
    % u1 = rot90(rot90(pic);
    u1 = zeros(N, N);
    delta = 10;
    u1((N-1)/2, [(N-1)/2 - delta, (N-1)/2 + delta]) = 1;
    
    z1 = 5*L;
    lad0 = L/100;
%     C = 1e3; lad0 = z1/C;
    f = 0.5*z1;
    R = 0.4*L;
    
    [u2, x2, y2] = ring_lens(u1, x1, y1, R, lad0, f, z1, 5);
    
    z2 = 1*z1*f/(z1 - f);
    [u3, x3, y3] = screen(u2, x2, y2, lad0, z2, L, 5);
    
    figure
    colormap('gray'); axis equal

    subplot(2, 2, 1)
    surf(x1, y1, abs(u1), 'edgecolor', 'none')
    view(0, 90); axis equal;

    subplot(2, 2, 2)
    surf(x2, y2, abs(u2), 'edgecolor', 'none')
    view(0, 90); axis equal;
    
    subplot(2, 2, 3)
    surf(x3, y3, abs(u3), 'edgecolor', 'none')
    view(0, 90); axis equal;
    
    figure
    colormap('gray'); axis equal

    subplot(1, 2, 1)
    surf(x1, y1, abs(u1), 'edgecolor', 'none')
    view(0, 90); axis equal;
    xlim([min(x1), max(x1)]); ylim([min(y1), max(y1)]);
    title('Input', 'interpreter', 'latex')
    xlabel('x', 'interpreter', 'latex'); ylabel('y', 'interpreter', 'latex');


    subplot(1, 2, 2)
    surf(x3, y3, abs(u3), 'edgecolor', 'none')
    view(0, 90); axis equal;
    xlim([min(x3), max(x3)]); ylim([min(y3), max(y3)]);
    title('Output', 'interpreter', 'latex')
    xlabel('x', 'interpreter', 'latex'); ylabel('y', 'interpreter', 'latex');


end
    
% PSF
if(0)
    L = 10; N = 201;
    x1 = linspace(-L/2, L/2, N); y1 = linspace(-L/2, L/2, N);
    [X1, Y1] = meshgrid(x1, y1);
    func_rect = @(x, y) 0 + 1.*((abs(x) <= 2) & (abs(y) <= 0.3));% 1.*((abs(x) <= 1) & (abs(y) <= 2));

    % pic = imread('picpic.png');
    % [Ynum, Xnum] = size(pic(:, :, 1));
    % Xnum = (Xnum - 1)*(mod(Xnum, 2) == 0) + Xnum*(mod(Xnum, 2) == 1);
    % Ynum = (Ynum - 1)*(mod(Ynum, 2) == 0) + Ynum*(mod(Ynum, 2) == 1);
    % x1 = -(Ynum - 1)/2:(Ynum - 1)/2;
    % y1 = -(Ynum - 1)/2:(Ynum - 1)/2;
    % % pic = (pic >= 200);
    % pic = pic(1:Ynum, 1:Ynum, 1);

    % u1 = pic;
    % u1 = rot90(rot90(pic);
    u1 = zeros(N, N);
    u1(10:25:190, 10:25:190) = 1;
    
    z1 = 7*L;
    lad0 = L/1000;
%     C = 1e3; lad0 = z1/C;
    f = 0.5*z1;
    R = 0.5*L;
    
    [u2, x2, y2] = ring_lens(u1, x1, y1, R, lad0, f, z1, 5);
    
    z2 = 1*z1*f/(z1 - f);
    [u3, x3, y3] = screen(u2, x2, y2, lad0, z2, L, 5);
    
    figure
    colormap('gray'); axis equal

    subplot(2, 2, 1)
    surf(x1, y1, abs(u1), 'edgecolor', 'none')
    view(0, 90); axis equal;

    subplot(2, 2, 2)
    surf(x2, y2, abs(u2), 'edgecolor', 'none')
    view(0, 90); axis equal;
    
    subplot(2, 2, 3)
    surf(x3, y3, abs(u3), 'edgecolor', 'none')
    view(0, 90); axis equal;
    
    figure
    colormap('gray'); axis equal

    subplot(1, 2, 1)
    surf(x1, y1, abs(u1), 'edgecolor', 'none')
    view(0, 90); axis equal;
    xlim([min(x1), max(x1)]); ylim([min(y1), max(y1)]);
    title('Input', 'interpreter', 'latex')
    xlabel('x', 'interpreter', 'latex'); ylabel('y', 'interpreter', 'latex');


    subplot(1, 2, 2)
    surf(x3, y3, abs(u3), 'edgecolor', 'none')
    view(0, 90); axis equal;
    xlim([min(x3), max(x3)]); ylim([min(y3), max(y3)]);
    title('Output', 'interpreter', 'latex')
    xlabel('x', 'interpreter', 'latex'); ylabel('y', 'interpreter', 'latex');

end

% Comma
if(0)
    L = 10; N = 201;
    x1 = linspace(-L/2, L/2, N); y1 = linspace(-L/2, L/2, N);
    [X1, Y1] = meshgrid(x1, y1);
    func_rect = @(x, y) 0 + 1.*((abs(x) <= 2) & (abs(y) <= 0.3));% 1.*((abs(x) <= 1) & (abs(y) <= 2));

    % pic = imread('picpic.png');
    % [Ynum, Xnum] = size(pic(:, :, 1));
    % Xnum = (Xnum - 1)*(mod(Xnum, 2) == 0) + Xnum*(mod(Xnum, 2) == 1);
    % Ynum = (Ynum - 1)*(mod(Ynum, 2) == 0) + Ynum*(mod(Ynum, 2) == 1);
    % x1 = -(Ynum - 1)/2:(Ynum - 1)/2;
    % y1 = -(Ynum - 1)/2:(Ynum - 1)/2;
    % % pic = (pic >= 200);
    % pic = pic(1:Ynum, 1:Ynum, 1);

    % u1 = pic;
    % u1 = rot90(rot90(pic);
    u1 = zeros(N, N);
    u1(20, 20) = 1;
    
    z1 = 7*L;
    lad0 = L/1000;
%     C = 1e3; lad0 = z1/C;
    f = 0.5*z1;
    R = 0.5*L;
    
    [u2, x2, y2] = ring_lens(u1, x1, y1, R, lad0, f, z1, 5);
    
    z2 = z1*f/(z1 - f);
    [u3, x3, y3] = screen(u2, x2, y2, lad0, z2, L, 5);
    
    figure
    colormap('gray'); axis equal

    subplot(2, 2, 1)
    surf(x1, y1, abs(u1), 'edgecolor', 'none')
    view(0, 90); axis equal;

    subplot(2, 2, 2)
    surf(x2, y2, abs(u2), 'edgecolor', 'none')
    view(0, 90); axis equal;
    
    subplot(2, 2, 3)
    surf(x3, y3, abs(u3), 'edgecolor', 'none')
    view(0, 90); axis equal;
    
    figure
    colormap('gray'); axis equal

    subplot(1, 2, 1)
    surf(x1, y1, abs(u1), 'edgecolor', 'none')
    view(0, 90); axis equal;
    xlim([min(x1), max(x1)]); ylim([min(y1), max(y1)]);
    title('Input', 'interpreter', 'latex')
    xlabel('x', 'interpreter', 'latex'); ylabel('y', 'interpreter', 'latex');


    subplot(1, 2, 2)
    surf(x3, y3, abs(u3), 'edgecolor', 'none')
    view(0, 90); axis equal;
    xlim([min(x3), max(x3)]); ylim([min(y3), max(y3)]);
    title('Output', 'interpreter', 'latex')
    xlabel('x', 'interpreter', 'latex'); ylabel('y', 'interpreter', 'latex');

end

% 4f System
if(0)
    L = 10; N = 201;
    x1 = linspace(-L/2, L/2, N); y1 = linspace(-L/2, L/2, N);
    [X1, Y1] = meshgrid(x1, y1);
    func_rect = @(x, y) 0 + 1.*((abs(x) <= 2) & (abs(y) <= 0.3));% 1.*((abs(x) <= 1) & (abs(y) <= 2));

    pic = imread('chaplinsq.jpg');
    pic = pic(:, :, 1);

    
    u1 = rot90(rot90(pic));
%     u1 = func_rect(X1, Y1 + 2);
    
    z1 = 10*L;
    lad0 = L/500;
    f = z1;
    R = 2*L;
    
    [u2, x2, y2] = ring_lens(u1, x1, y1, R, lad0, f, z1, 3);
    
    z2 = 2*f;
    [u3, x3, y3] = ring_lens(u2, x2, y2, R, lad0, f, z2, 3);

    
    z3 = f;
    [u4, x4, y4] = screen(u3, x3, y3, lad0, z3, L, 3);
    u4 = u4/max(abs(u4(:)));
    
    figure
    colormap('gray'); axis equal

    subplot(2, 2, 1)
    surf(x1, y1, abs(u1), 'edgecolor', 'none')
    view(0, 90); axis equal;

    subplot(2, 2, 2)
    surf(x2, y2, abs(u2), 'edgecolor', 'none')
    view(0, 90); axis equal;
    
    subplot(2, 2, 3)
    surf(x3, y3, abs(u3), 'edgecolor', 'none')
    view(0, 90); axis equal;

    subplot(2, 2, 4)
    surf(x4, y4, abs(u4), 'edgecolor', 'none')
    view(0, 90); axis equal;
    
    figure
    colormap('gray'); axis equal

    subplot(1, 2, 1)
    surf(x1, y1, abs(u1), 'edgecolor', 'none')
    view(0, 90); axis equal;
    xlim([min(x1), max(x1)]); ylim([min(y1), max(y1)]);
    title('Input', 'interpreter', 'latex')
    xlabel('x', 'interpreter', 'latex'); ylabel('y', 'interpreter', 'latex');


    subplot(1, 2, 2)
    surf(x4, y4, abs(u4), 'edgecolor', 'none')
    view(0, 90); axis equal;
    xlim([min(x4), max(x4)]); ylim([min(y4), max(y4)]);
    title('Output', 'interpreter', 'latex')
    xlabel('x', 'interpreter', 'latex'); ylabel('y', 'interpreter', 'latex');

end



% Low Pass
if(0)
    L = 10; N = 201;
    x1 = linspace(-L/2, L/2, N); y1 = linspace(-L/2, L/2, N);
    [X1, Y1] = meshgrid(x1, y1);
    func_rect = @(x, y) 0 + 1.*((abs(x) <= 2) & (abs(y) <= 0.3));% 1.*((abs(x) <= 1) & (abs(y) <= 2));

    pic = imread('chaplinsq.jpg');
    pic = pic(:, :, 1);

    
    u1 = rot90(rot90(pic));
    
    z1 = 10*L;
    lad0 = L/100;
%     C = 10e3; lad0 = z1/C;
    f = z1;
    R = 2*L;
    
    [u2, x2, y2] = ring_lens(u1, x1, y1, R, lad0, f, z1, 3);
    u2 = u2/max(abs(u2(:)));

    zz = f;
    R_app = R;
    [uu, xx, yy] = ring_apperture(u2, x2, y2, R_app, lad0, zz, 3);
    uu = uu/max(abs(uu(:)));

    z2 = f;
    [u3, x3, y3] = ring_lens(uu, xx, yy, R, lad0, f, z2, 3);
    u3 = u3/max(abs(u3(:)));

    
    z3 = f;
    [u4, x4, y4] = screen(u3, x3, y3, lad0, z3, L, 3);
    u4 = u4/max(abs(u4(:)));
    
    figure
    colormap('gray'); axis equal

    subplot(2, 3, 1)
    surf(x1, y1, abs(u1), 'edgecolor', 'none')
    view(0, 90); axis equal;

    subplot(2, 3, 2)
    surf(x2, y2, abs(u2), 'edgecolor', 'none')
    view(0, 90); axis equal;

    subplot(2, 3, 3)
    surf(xx, yy, abs(uu), 'edgecolor', 'none')
    view(0, 90); axis equal;

    
    subplot(2, 3, 4)
    surf(x3, y3, abs(u3), 'edgecolor', 'none')
    view(0, 90); axis equal;

    subplot(2, 3, 5)
    surf(x4, y4, abs(u4), 'edgecolor', 'none')
    view(0, 90); axis equal;
    
    figure
    colormap('gray'); axis equal

    subplot(1, 2, 1)
    surf(x1, y1, abs(u1), 'edgecolor', 'none')
    view(0, 90); axis equal;
    xlim([min(x1), max(x1)]); ylim([min(y1), max(y1)]);
    title('Input', 'interpreter', 'latex')
    xlabel('x', 'interpreter', 'latex'); ylabel('y', 'interpreter', 'latex');


    subplot(1, 2, 2)
    surf(x4, y4, abs(u4), 'edgecolor', 'none')
    view(0, 90); axis equal;
    xlim([min(x4), max(x4)]); ylim([min(y4), max(y4)]);
    title('Output', 'interpreter', 'latex')
    xlabel('x', 'interpreter', 'latex'); ylabel('y', 'interpreter', 'latex');

end

% High Pass
if(1)
    L = 10; N = 201;
    x1 = linspace(-L/2, L/2, N); y1 = linspace(-L/2, L/2, N);
    [X1, Y1] = meshgrid(x1, y1);
    func_rect = @(x, y) 0 + 1.*((abs(x) <= 2) & (abs(y) <= 0.3));% 1.*((abs(x) <= 1) & (abs(y) <= 2));

    pic = imread('chaplinsq.jpg');
    pic = pic(:, :, 1);

    
    u1 = rot90(rot90(pic));
    
    z1 = 10*L;
    C = 10e3; lad0 = z1/C;
    f = z1;
    R = 2*L;
    
    [u2, x2, y2] = ring_lens(u1, x1, y1, R, lad0, f, z1, 3);
    u2 = u2/max(abs(u2(:)));

    zz = f;
    R1 = R/5;
    R2 = R;
    [uu, xx, yy] = disk_apperture(u2, x2, y2, R1, R2, lad0, zz, 3);
    uu = uu/max(abs(uu(:)));

    z2 = f;
    [u3, x3, y3] = ring_lens(uu, xx, yy, R, lad0, f, z2, 3);
    u3 = u3/max(abs(u3(:)));

    
    z3 = f;
    [u4, x4, y4] = screen(u3, x3, y3, lad0, z3, L, 3);
    u4 = u4/max(abs(u4(:)));
    
    figure
    colormap('gray'); axis equal

    subplot(2, 3, 1)
    surf(x1, y1, abs(u1), 'edgecolor', 'none')
    view(0, 90); axis equal;

    subplot(2, 3, 2)
    surf(x2, y2, abs(u2), 'edgecolor', 'none')
    view(0, 90); axis equal;

    subplot(2, 3, 3)
    surf(xx, yy, abs(uu), 'edgecolor', 'none')
    view(0, 90); axis equal;

    
    subplot(2, 3, 4)
    surf(x3, y3, abs(u3), 'edgecolor', 'none')
    view(0, 90); axis equal;

    subplot(2, 3, 5)
    surf(x4, y4, abs(u4), 'edgecolor', 'none')
    view(0, 90); axis equal;
    
    figure
    colormap('gray'); axis equal

    subplot(1, 2, 1)
    surf(x1, y1, abs(u1), 'edgecolor', 'none')
    view(0, 90); axis equal;
    xlim([min(x1), max(x1)]); ylim([min(y1), max(y1)]);
    title('Input', 'interpreter', 'latex')
    xlabel('x', 'interpreter', 'latex'); ylabel('y', 'interpreter', 'latex');


    subplot(1, 2, 2)
    surf(x4, y4, abs(u4), 'edgecolor', 'none')
    view(0, 90); axis equal;
    xlim([min(x4), max(x4)]); ylim([min(y4), max(y4)]);
    title('Output', 'interpreter', 'latex')
    xlabel('x', 'interpreter', 'latex'); ylabel('y', 'interpreter', 'latex');

end

% Perfect Lens
if(0)
    
end

% Multi Lens
if(0)

end

% Telescope
if(0)

end

% Black Hole
if(0)

end

% Quantum Light
if(0)
    L = 10; N = 301;
    x1 = linspace(-L/2, L/2, N); y1 = linspace(-L/2, L/2, N);
    [X1, Y1] = meshgrid(x1, y1);

    func_rect = @(x, y) 0 + 1.*((x.^2 + y.^2) <= 0.2);% 1.*((abs(x) <= 1) & (abs(y) <= 2));

    u1 = func_rect(X1, 0.75*Y1);

    z1 = 25*L;
    C = 5e3; lad0 = z1/C;
    D = 10*L;
    [u2, x2, y2] = screen(u1, x1, y1, lad0, z1, D, 3);

    figure
    colormap('gray'); axis equal

    subplot(1, 2, 1)
    surf(x1, y1, abs(u1), 'edgecolor', 'none')
    view(0, 90); axis equal;

    subplot(1, 2, 2)
    surf(x2, y2, abs(u2), 'edgecolor', 'none')
    view(0, 90); axis equal;

    prob_distribution = abs(u2);
    x = x2; y = y2;
    L = max(x) - min(x); N = length(x);
    [X, Y] = meshgrid(x, y);
    
    nbar = 10;
    
    test_num = 2000;
    [x0, y0] = sampling(prob_distribution, x, y, test_num);
    
    photon_intensity = zeros(N, N);
    photon_size = L/60;

    for i = 1:test_num
        n = poissrnd(nbar);
        photon_intensity = photon_intensity + exp(-((X - x0(i)).^2 + (Y - y0(i)).^2) / (photon_size^2*(n/nbar)));

    end


    figure
    colormap('gray');
    surf(x, y, photon_intensity, 'edgecolor', 'none')
    view(0, 90); axis equal;    % scatter(x0, y0, 3, 'filled', 'r'); axis equal;colormap('jet');
    

    
end


function [x0, y0] = sampling(P, x, y, q)
    % Sampling from a 2D probability distribution P(x, y).
    %
    % Inputs:
    % P: Probability matrix (N x N).
    % x: X coordinates of the meshgrid.
    % y: Y coordinates of the meshgrid.
    % q: Number of samples to generate.
    %
    % Outputs:
    % x0: Array of sampled x coordinates (1 x q).
    % y0: Array of sampled y coordinates (1 x q).

    % Normalize P to create a valid probability distribution.
    P = P / sum(P(:));

    % Flatten P and create a corresponding grid of indices.
    P_flat = P(:); % Flatten P into a column vector.
    [X, Y] = meshgrid(x, y); % Generate meshgrid of x and y.
    X_flat = X(:); % Flatten X into a column vector.
    Y_flat = Y(:); % Flatten Y into a column vector.

    % Sample q indices based on the probability distribution.
    sample_indices = randsample(1:length(P_flat), q, true, P_flat);

    % Get the sampled (x, y) coordinates.
    x0 = X_flat(sample_indices);
    y0 = Y_flat(sample_indices);
end

function out = point_creator(x0, y0, x, y)
    [X, Y] = meshgrid(x, y);
    out = zeros(length(y), length(x));
    out(X == x0 & Y == y0) = 1;

end

function [out, x_new, y_new] = free_space(lad0, z, u_in, x, y, pad_factor)
    Lx = max(x) - min(x); Ly = max(y) - min(y);
    [y_num, x_num] = size(u_in);
    scaling = 1 + 2*pad_factor;
    N_pad = pad_factor*max(x_num, y_num);
    u_pad = padarray(u_in, [N_pad, N_pad], 0, 'both');
    x_new = linspace(-Lx/2*scaling, Lx/2*scaling, x_num + 2*N_pad);
    y_new = linspace(-Ly/2*scaling, Ly/2*scaling, y_num + 2*N_pad);
    [X_new, Y_new] = meshgrid(x_new, y_new);
    h = exp(1i*2*pi*z/lad0)/(1i*lad0*z)*exp(1i*pi/lad0/z*(X_new.^2 + Y_new.^2));
%     h = exp(1i*pi/lad0/z*(X_new.^2 + Y_new.^2));
    
    F_u_pad = (fft2(u_pad));
%     F_u_pad = F_u_pad./max(abs(F_u_pad(:)));
    F_h = (fft2(h));
%     F_h = F_h./max(abs(F_h(:)));

    out = fftshift(ifft2(F_h.*F_u_pad));
    out = out./max(abs(out(:)));

    
    
end

function u_out = high_pass(u_in, x, y, factor)
    R = factor*(max(x) - min(x))/2;
    [X, Y] = meshgrid(x, y);
    filter = zeros(length(x));
    filter = ((X.^2 + Y.^2) > R^2);

    u_out = ifft2(filter.*fftshift(fft2(u_in)));
    

end

function [u_out, x_out, y_out] = object_circ(u_in, x_in, y_in, R, lad0, z, pad_factor)
    L = max(x_in) - min(x_in);
    N = size(u_in, 1);
           
    scaling = 1 + 2*pad_factor;
    
    N_pad = ceil(pad_factor*N);
    u_pad = padarray(u_in, [N_pad, N_pad], 0, 'both');
    x2 = linspace(-L/2*scaling, L/2*scaling, N + 2*N_pad);
    y2 = linspace(-L/2*scaling, L/2*scaling, N + 2*N_pad);
    [X2, Y2] = meshgrid(x2, y2);
    h = exp(1i*2*pi*z/lad0)/(1i*lad0*z)*exp(1i*pi/lad0/z*(X2.^2 + Y2.^2));
    
    F_u_pad = (fft2(u_pad));
    F_h = (fft2(h));

    u2 = fftshift(ifft2(F_h.*F_u_pad));
    u2 = u2./max(abs(u2(:)));

    circ = @(u, v) 0 + 1.*(sqrt((u - R).^2 + (v).^2) <= R);
    x_out = x2;
    y_out = y2;
    u_out = u2 - u2.*circ(X2, Y2) + circ(X2, Y2);

end

function [u_out, x_out, y_out] = ring_apperture(u_in, x_in, y_in, R, lad0, z, pad_factor)
    
    if(z < 2*(2*R).^2/lad0)
        L = max(x_in) - min(x_in);
        N = size(u_in, 1);
               
        while((2*R) > (L*(1 + 2*pad_factor)))
            pad_factor = pad_factor + 0.5;
        end
        
        scaling = 1 + 2*pad_factor;
        
        N_pad = ceil(pad_factor*N);
        u_pad = padarray(u_in, [N_pad, N_pad], 0, 'both');
        x2 = linspace(-L/2*scaling, L/2*scaling, N + 2*N_pad);
        y2 = linspace(-L/2*scaling, L/2*scaling, N + 2*N_pad);
        [X2, Y2] = meshgrid(x2, y2);
        h = exp(1i*2*pi*z/lad0)/(1i*lad0*z)*exp(1i*pi/lad0/z*(X2.^2 + Y2.^2));
        
        F_u_pad = (fft2(u_pad));
        F_h = (fft2(h));
    
        u2 = fftshift(ifft2(F_h.*F_u_pad));
    
        t_apperture = @(u, v) 0 + 1.*(sqrt((u).^2 + (v).^2) <= R);
        % x_out = linspace(-R, R, N);
        % y_out = linspace(-R, R, N);
        u_out = u2.*t_apperture(X2, Y2);

        I = find(abs(x2) < R);

        x_out = x2(I);
        y_out = y2(I);
        u_out = u_out(I, I);
        u_out = u_out./max(abs(u_out(:)));

        33

    else
        N = length(x_in);
        L = max(x_in) - min(x_in);
        
        k0 = 2*pi/lad0;

        [X_in, Y_in] = meshgrid(x_in, y_in);

        Kx = -k0*X_in/z;
        Ky = -k0*Y_in/z;
        kx = Kx(1, :);
        
        F_u = rot90(rot90(u_in));
        

        N_pad = ceil(pad_factor*N);
        F_u_pad = padarray(F_u, [N_pad, N_pad], 0, 'both');

        dkx = kx(2) - kx(1); % Sampling interval in kx
        dky = dkx; % Sampling interval in ky
    
        % Spatial grid size
        x_out = linspace(-pi/dkx, pi/dkx, N + 2*N_pad); % x-coordinates
        y_out = linspace(-pi/dky, pi/dky, N + 2*N_pad); % y-coordinates
        
        u_out = abs(ifftshift(ifft2((F_u_pad)))); 

        % I = find(abs(x_out)<(D/2))


    end
end

function [u_out, x_out, y_out] = disk_apperture(u_in, x_in, y_in, R1, R2, lad0, z, pad_factor)
    
    if(z < 2*(2*R2).^2/lad0)
        L = max(x_in) - min(x_in);
        N = size(u_in, 1);
               
        while((2*R2) > (L*(1 + 2*pad_factor)))
            pad_factor = pad_factor + 0.5;
        end
        
        scaling = 1 + 2*pad_factor;
        
        N_pad = ceil(pad_factor*N);
        u_pad = padarray(u_in, [N_pad, N_pad], 0, 'both');
        x2 = linspace(-L/2*scaling, L/2*scaling, N + 2*N_pad);
        y2 = linspace(-L/2*scaling, L/2*scaling, N + 2*N_pad);
        [X2, Y2] = meshgrid(x2, y2);
        h = exp(1i*2*pi*z/lad0)/(1i*lad0*z)*exp(1i*pi/lad0/z*(X2.^2 + Y2.^2));
        
        F_u_pad = (fft2(u_pad));
        F_h = (fft2(h));
    
        u2 = fftshift(ifft2(F_h.*F_u_pad));
    
        t_apperture = @(u, v) 0 + 1.*(sqrt((u).^2 + (v).^2) >= R1).*((sqrt((u).^2 + (v).^2) <= R2));
        % x_out = linspace(-R, R, N);
        % y_out = linspace(-R, R, N);
        u_out = u2.*t_apperture(X2, Y2);

        I = find(abs(x2) < R2);

        x_out = x2(I);
        y_out = y2(I);
        u_out = u_out(I, I);
        u_out = u_out./max(abs(u_out(:)));

        33

    else
        N = length(x_in);
        L = max(x_in) - min(x_in);
        
        k0 = 2*pi/lad0;

        [X_in, Y_in] = meshgrid(x_in, y_in);

        Kx = -k0*X_in/z;
        Ky = -k0*Y_in/z;
        kx = Kx(1, :);
        
        F_u = rot90(rot90(u_in));
        

        N_pad = ceil(pad_factor*N);
        F_u_pad = padarray(F_u, [N_pad, N_pad], 0, 'both');

        dkx = kx(2) - kx(1); % Sampling interval in kx
        dky = dkx; % Sampling interval in ky
    
        % Spatial grid size
        x_out = linspace(-pi/dkx, pi/dkx, N + 2*N_pad); % x-coordinates
        y_out = linspace(-pi/dky, pi/dky, N + 2*N_pad); % y-coordinates
        
        u_out = abs(ifftshift(ifft2((F_u_pad)))); 

        % I = find(abs(x_out)<(D/2))


    end
end


function [u_out, x_out, y_out] = ring_lens(u_in, x_in, y_in, R, lad0, f, z, pad_factor)

    if(z < 2*(2*R).^2/lad0)

        L = max(x_in) - min(x_in);
        N = size(u_in, 1);
        
        
        while((2*R) > (L*(1 + 2*pad_factor)))
            pad_factor = pad_factor + 0.5;
        end
        
        scaling = 1 + 2*pad_factor;
        
        N_pad = ceil(pad_factor*N);
        u_pad = padarray(u_in, [N_pad, N_pad], 0, 'both');
        x2 = linspace(-L/2*scaling, L/2*scaling, N + 2*N_pad);
        y2 = linspace(-L/2*scaling, L/2*scaling, N + 2*N_pad);
        [X2, Y2] = meshgrid(x2, y2);
        h = exp(1i*2*pi*z/lad0)/(1i*lad0*z)*exp(1i*pi/lad0/z*(X2.^2 + Y2.^2));
        pad_factor
        
        F_u_pad = (fft2(u_pad));
        F_h = (fft2(h));
    
        u2 = fftshift(ifft2(F_h.*F_u_pad));
    
        t_lens = @(u, v) 0 + 1.*exp(-1i*pi/lad0/f*(u.^2 + v.^2)).*(sqrt(u.^2 + v.^2) <= R);
        u_out = u2.*t_lens(X2, Y2);
    
        I = find(abs(x2) < R);
    
        x_out = x2(I);
        y_out = y2(I);
        u_out = u_out(I, I);
        u_out = u_out./max(abs(u_out(:)));


    else
        N = length(x_in);
        L = max(x_in) - min(x_in);
        
        k0 = 2*pi/lad0;

        [X_in, Y_in] = meshgrid(x_in, y_in);

        Kx = -k0*X_in/z;
        Ky = -k0*Y_in/z;
        kx = Kx(1, :);
        
        F_u = rot90(rot90(u_in));
        

        N_pad = ceil(pad_factor*N);
        F_u_pad = padarray(F_u, [N_pad, N_pad], 0, 'both');

        dkx = kx(2) - kx(1); % Sampling interval in kx
        dky = dkx; % Sampling interval in ky
    
        % Spatial grid size
        x_out = linspace(-pi/dkx, pi/dkx, N + 2*N_pad); % x-coordinates
        y_out = linspace(-pi/dky, pi/dky, N + 2*N_pad); % y-coordinates
        
        u_out = abs(ifftshift(ifft2((F_u_pad)))); 

        % I = find(abs(x_out)<(D/2))

    end
    
end

function [u_out, x_out, y_out] = screen(u_in, x_in, y_in, lad0, z, D, pad_factor)
    % D is the length of the square side of the screen
    if(z < 2*D.^2/lad0)
        L = max(x_in) - min(x_in);
        N = size(u_in, 1);

        while(D > (L*(1 + 2*pad_factor)))
            pad_factor = pad_factor + 0.5;
        end
        scaling = 1 + 2*pad_factor;

        N_pad = ceil(pad_factor*N);
        u_pad = padarray(u_in, [N_pad, N_pad], 0, 'both');
        x_out = linspace(-L/2*scaling, L/2*scaling, N + 2*N_pad);
        y_out = linspace(-L/2*scaling, L/2*scaling, N + 2*N_pad);
        [X_out, Y_out] = meshgrid(x_out, y_out);
        h = exp(1i*2*pi*z/lad0)/(1i*lad0*z)*exp(1i*pi/lad0/z*(X_out.^2 + Y_out.^2));
    %     h = exp(1i*pi/lad0/z*(X_new.^2 + Y_new.^2));

        F_u_pad = (fft2(u_pad));
    %     F_u_pad = F_u_pad./max(abs(F_u_pad(:)));
        F_h = (fft2(h));
    %     F_h = F_h./max(abs(F_h(:)));

        u_out = fftshift(ifft2(F_h.*F_u_pad));
    %     u_out = out./max(abs(out(:)));

        I = find(abs(x_out)<(D/2));

        x_out = x_out(I);
        y_out = y_out(I);
        u_out = u_out(I, I);
        u_out = u_out./max(abs(u_out(:)));
        33
    
    else
        N = length(x_in);
        L = max(x_in) - min(x_in);
        
        k0 = 2*pi/lad0;

        [X_in, Y_in] = meshgrid(x_in, y_in);

        Kx = -k0*X_in/z;
        Ky = -k0*Y_in/z;
        kx = Kx(1, :);
        
        F_u = rot90(rot90(u_in));
        

        N_pad = ceil(pad_factor*N);
        F_u_pad = padarray(F_u, [N_pad, N_pad], 0, 'both');

        dkx = kx(2) - kx(1); % Sampling interval in kx
        dky = dkx; % Sampling interval in ky
    
        % Spatial grid size
        x_out = linspace(-pi/dkx, pi/dkx, N + 2*N_pad); % x-coordinates
        y_out = linspace(-pi/dky, pi/dky, N + 2*N_pad); % y-coordinates
        
        u_out = abs(ifftshift(ifft2((F_u_pad))));
        
        % [X_in, Y_in] = meshgrid(x_in, y_in);
        % Kx = -k0*X_in/z;
        % Ky = -k0*Y_in/z;
        
        % replicator = ones(interpolate, interpolate);
        % F_u = rot90(rot90(kron(u_in, replicator)));
        % 
        % 
        % xx = linspace(-L/2, L/2, N*interpolate);
        % yy = linspace(-L/2, L/2, N*interpolate);
        % [XX, YY] = meshgrid(xx, yy);
        % Kx = -k0*XX/z;
        % Ky = -k0*YY/z;
        % 
        % dkx = Kx(1, 2) - Kx(1, 1); dky = dkx;
        % 
        % x_out = linspace(-pi/dkx, pi/dkx, N*interpolate);
        % y_out = linspace(-pi/dky, pi/dky, N*interpolate);
        % 
        % u_out = abs(ifftshift(ifft2((F_u))));
        % % 
        % % I = find(abs(xx)<(D/2));
        % % 
        % % x_out = x_out(I);
        % % y_out = y_out(I);
        % % u_out = u_out(I, I);
        % 
        % % u_out = F_u;

        
        
    end
    
    
end