library(caret)
library(mgcv)
expit=function(x){
        return(exp(x)/(1+exp(x)))
}

ns=100
K=5
mn=1
model_name="LM"
model_name="GAM"

fno=paste("Book/m",mn,"_",model_name,".txt", sep="")
for(i in 1:ns){
	fn=paste("Book/m", mn, "/",i,".csv",sep="")
	data=read.csv(fn)
	Y_res=data$Y[data$r_idx==1]
	X_res=data[data$r_idx==1,c(2:7)]
	X=data[,c(2:7)]
	Y=data$Y
	X_mis=data[data$r_idx!=1,c(2:7)]
	rd=data.frame(Y_res, X_res)

	folds=createFolds(data$r_idx, k=K, list=T)
	#1. Linear regression model
	if(model_name == "LM"){
		fit=lm(Y_res~., data=rd)
		fit1_resp = glm(r_idx~., wres[,-1], family="binomial")
	}
	
	if(model_name == "GAM"){
		f <- reformulate(setdiff(colnames(rd), "Y_res"), response="Y_res")
		fit = gam(f, data = rd)
		f <- reformulate(setdiff(colnames(data[,-1]), "r_idx"), response="r_idx")
		fit1_resp = gam(f, data[,-1], family="binomial")
	}

	Y_m = predict(fit, data.frame(X_mis))
	Y_m_c = predict(fit, data.frame(X))

	##Method 1 Simple imputation
	Y_est_1=(sum(Y_res)+sum(Y_m))/N

	##Method 2 Propensity score
	pv=predict(fit1_resp, data.frame(X))
	Y_est_2=sum(r_idx/expit(pv)*Y)/N

	##Method 3 Double machine learning
	mu_estimates <- numeric(K)  # Store estimates from each fold
	for (k in 1:K) {
  		# Define training and test sets
  		train_idx <- unlist(folds[-k])  # Use K-1 folds for training
 		test_idx <- folds[[k]]  # Use 1 fold for testing
  		train_data <- data[train_idx, ]
  		test_data <- data[test_idx, ]
  		# Train nuisance functions
		g_model_data=train_data[train_data$r_idx==1,-8]
		e_model_data=train_data[,-1]
		if(model_name == "LM"){
  			g_model <- lm(Y ~ ., data = g_model_data)  # Outcome model
  			e_model <- glm(r_idx ~ ., data = e_model_data, family="binomial")  # Propensity score model
		}
		
		if(model_name == "GAM"){
			f <- reformulate(setdiff(colnames(g_model_data), "Y"), response="Y")
  			g_model <- gam(f, data = g_model_data)  # Outcome model
			f <- reformulate(setdiff(colnames(e_model_data), "r_idx"), response="r_idx")
  			e_model <- gam(f, data = e_model_data, family="binomial")  # Propensity score model
		}

  		# Predict on test set
		test_data_g=test_data[,-8]
  		g_X_test <- predict(g_model, data.frame(test_data_g))
		test_data_e=test_data[,-1]
  		e_X_test <- predict(e_model, data.frame(test_data_e), type="response")

  		# Compute doubly robust estimate on test set

  		mu_estimates[k] <- mean(g_X_test + (test_data$r_idx / e_X_test) * (test_data$Y - g_X_test))

	}

	Y_est_3 <- mean(mu_estimates)
	write.table(data.frame(Y_est_1,Y_est_2,Y_est_3),fno,append=T,row.names=F,col.names=F,sep=",")
}
