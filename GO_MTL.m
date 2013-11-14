function [L,S,W] = GO_MTL(X_train,Y_train,k,Maxsteps,lambda,mu,O)
   
    T = size(X_train,1);
    D = size(X_train{1},2);
    W = zeros(D, T);
    mfOptions.Method = 'newton';
    options.verbose = false;
    
    if O == 'R'
        baseObj = @SquaredError;
    else
        baseObj = @LogisticLoss;
    end

    for t = 1:T
        funObj = @(w)baseObj(w,X_train{t},Y_train{t});
        funObjL2 = @(w)penalizedL2(w,funObj,lambda*ones(D,1));
        W(:,t) = minFunc(funObjL2,zeros(D,1),mfOptions);
    end
        
    [U,~,~] = svd(W);
    L = U(:,1:k);
    S = zeros(k, T);

    for step = 1:Maxsteps
        for t = 1:T
            funObj = @(s)baseObj(s,X_train{t}*L,Y_train{t});
            S(:,t) = L1General2_PSSgb(funObj,S(:,t),mu*ones(k,1),options);
        end
        if O == 'R'
            B = zeros(D*k,1);
            A = eye(D*k) * lambda;
            for t = 1:T
                N_t = size(X_train{t},1);
                A = A + 1/N_t*kron(S(:,t)*S(:,t)',X_train{t}'*X_train{t});
                b = X_train{t}'*Y_train{t}*S(:,t)';
                B = B + 1/N_t*b(:);
            end
            L = reshape(linsolve(A,B),size(L));
        else
            deltaL = 2*lambda*L;
            for t = 1:T
                N_t = size(X_train{t},1);
                deltaL = deltaL - 1/N_t*((diag((Y_train{t}-1./(1+exp(-X_train{t}*L*S(:,t)))))*X_train{t})'*repmat(S(:,t)',[N_t,1]));
            end
            L = L - 0.005*deltaL;
        end
    end
end