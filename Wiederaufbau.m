function [Wsignal] = Wiederaufbau(Tx,GratlinieIndex,modn,sigma,Delta)
% Tx ist die Darstellung des Transformation.  Tx是时频表示。 Tx is the TFR.
%     GratlinieIndex 是脊线。
%     modn ist mal der Wiederaufbau.  modn是重构次数。 modn is the number for reconstruction.
%     sigma ist das Koeffizient der Fensterweite.  sigma是窗宽系数。sigma of gaussian window
%     Delta ist "r". 文章中的"r". "r" in the paper.

%     @Autor: Shaowen Chen
%     2020.12.15

if (nargin < 2),
    error('Zahl und Gratlinieindex müssen eingegeben werden.');
end;
if (nargin < 3),
    error('Zahl muss eingegeben werden.');
end;
if (nargin < 4),
    sigma = 0.008;
end;
if (nargin < 5),
    Delta = 20;
end;



[Reihe,Spalte]= size(Tx);
for u=1:Spalte
    for i=1:modn
        summ = Tx(max(1,(GratlinieIndex(i,u)-Delta)):min(Reihe,(GratlinieIndex(i,u)+Delta)),u);        
        Wsignal(i,u) = real(sum(summ)/((pi*sigma^2)^(-0.25) *exp(-(0/sigma).^2/2)));
        Tx(max(1,(GratlinieIndex(i,u)-Delta)):min(Reihe,(GratlinieIndex(i,u)+Delta)),u) = zeros(1,length(summ));
    end
end

end

