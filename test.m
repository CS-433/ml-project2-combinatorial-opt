clc
N = 5;
A = rand(N);
s = rand(N,1);
s(s>0.5)=1;
s(s<=0.5)=-1;
for i = 1:N
    for j = 1:i
        A(i,j)=0;
    end
end
%A = A'+A
s'*A*s
eig(A)
        