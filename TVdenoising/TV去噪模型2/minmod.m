function c=minmod(a,b)
c=(sign(a)./2+sign(b)./2).*min(abs(a),abs(b));
end