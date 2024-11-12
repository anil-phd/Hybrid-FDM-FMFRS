function T = Smin(x)
    if (x >= 1)
        T = 1;
    else
        T = x;
    end
end

% function opVal = DSimRar(Ov_r,lpt,rpt)      % checking similiarity relation.
%     opVal = Ov_r / (Ov_r + ((lpt + rpt)/2));
% end
% 
% function opVal = OBSimRar(o_p,lpt,rpt)     % checking similiarity relation.
%     % 1 - (Ov_r / (Ov_r + lpt + rpt)); 
%     opVal = min((Ov_r/(Ov_r + lpt)),(Ov_r/(Ov_r + rpt)));
% end