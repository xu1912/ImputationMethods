library(caret)
library(parsnip)
library(workflows)
library(recipes)
library(dials)
library(tune)
library(rsample)
expit=function(x){
        return(exp(x)/(1+exp(x)))
}

N=500
ns=100
K=5
mn=1              ##indicates the data generation model 1-3

model_name="XGB"
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
	rdp=data.frame(data[,-1])
	rdp$r_idx=factor(rdp$r_idx, levels=c(0,1), labels=c("N","Y"))
	folds=createFolds(data$r_idx, k=K, list=T)

	common_recipe <- recipe(
		Y_res ~ ., data = rd
	)

	if(model_name == "XGB"){
      	# * 1) model spec rpart ----
		boost_spec <- boost_tree(
        		tree_depth = tune(),
        		trees = tune(),              # add tune() here as well - 20, 50, 100, 200, 300, etc.
        		learn_rate = tune(),
        		mtry = tune(),
  			min_n = tune(),
                	sample_size = 0.8
      	) %>%
		set_mode("regression") %>%
		set_engine("xgboost")

      	# * 2) workflow ----
		boost_workflow <- workflow() %>%
			add_recipe(common_recipe) %>%
			add_model(boost_spec)

      	# * 3) resamples ----
		set.seed(123)
		dat_folds <- vfold_cv(rd,v = 10, strata=sex)

      	# * 4) tune_grid() ----
      	set.seed(123)
      	boost_cube <- grid_latin_hypercube(
        		tree_depth(),
        		trees(),
        		learn_rate(),
        		finalize(mtry(), rd),
        		min_n(),
        		size = 30
      	)
      	boost_grid <- tune_grid(
        		boost_workflow,
        		resamples = dat_folds,
        		grid = boost_cube,
			control=control_grid(save_pred=T)
      	)

		final_boost_workflow <- boost_workflow %>%
		finalize_workflow(select_best(boost_grid,metric = "rmse"))

      	# * 6) last_fit() ----
      	fit <- extract_spec_parsnip(final_boost_workflow) %>%
       	fit(Y_res ~ ., data = rd)
	}

	Y_m = predict(fit, data.frame(X_mis))
	Y_m_c = predict(fit, data.frame(X))

	##Method 1
	Y_est_1=(sum(Y_res)+sum(Y_m))/N

	##Method 2
	common_recipe2 <- recipe(
		r_idx ~ ., data = rdp
	)

	if(model_name == "XGB"){
      	# * 1) model spec rpart ----
		boost_spec <- boost_tree(
        		tree_depth = tune(),
        		trees = tune(),              # add tune() here as well - 20, 50, 100, 200, 300, etc.
        		learn_rate = tune(),
        		mtry = tune(),
  			min_n = tune(),
                	sample_size = 0.8
      	) %>%
		set_mode("classification") %>%
		set_engine("xgboost")

      	# * 2) workflow ----
		boost_workflow <- workflow() %>%
			add_recipe(common_recipe2) %>%
			add_model(boost_spec)

      	# * 3) resamples ----
		set.seed(123)
		dat_folds <- vfold_cv(rdp,v = 10)

      	# * 4) tune_grid() ----
      	set.seed(123)
      	boost_cube <- grid_latin_hypercube(
        		tree_depth(),
        		trees(),
        		learn_rate(),
        		finalize(mtry(), rdp),
        		min_n(),
        		size = 30
      	)
      	boost_grid <- tune_grid(
        		boost_workflow,
        		resamples = dat_folds,
        		grid = boost_cube,
			control=control_grid(save_pred=T)
      	)

		final_boost_workflow <- boost_workflow %>%
		finalize_workflow(select_best(boost_grid,metric = "accuracy"))

      	# * 6) last_fit() ----
      	fit1_resp <- extract_spec_parsnip(final_boost_workflow) %>%
       	fit(r_idx ~ ., data = rdp)

	}
	pv=predict(fit1_resp, data.frame(X),type="prob")
	pv$.pred_Y[pv$.pred_Y==0]=0.01
	Y_est_2=sum(data$r_idx/pv$.pred_Y*Y)/N

	##Method 3
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
		if(model_name == "XGB"){
			# Outcome model

			common_recipe <- recipe(
				Y ~ ., data = g_model_data
			)

      		# * 1) model spec rpart ----
			boost_spec <- boost_tree(
        			tree_depth = tune(),
        			trees = tune(),              # add tune() here as well - 20, 50, 100, 200, 300, etc.
        			learn_rate = tune(),
        			mtry = tune(),
  				min_n = tune(),
                		sample_size = 0.8
      		) %>%
			set_mode("regression") %>%
			set_engine("xgboost")

      		# * 2) workflow ----
			boost_workflow <- workflow() %>%
				add_recipe(common_recipe) %>%
				add_model(boost_spec)

      		# * 3) resamples ----
			set.seed(123)
			dat_folds <- vfold_cv(g_model_data,v = 10)

      		# * 4) tune_grid() ----
      		set.seed(123)
      		boost_cube <- grid_latin_hypercube(
        			tree_depth(),
        			trees(),
        			learn_rate(),
        			finalize(mtry(), g_model_data),
        			min_n(),
        			size = 30
      		)
      		boost_grid <- tune_grid(
        			boost_workflow,
        			resamples = dat_folds,
        			grid = boost_cube,
				control=control_grid(save_pred=T)
      		)


			final_boost_workflow <- boost_workflow %>%
			finalize_workflow(select_best(boost_grid,metric = "rmse"))

      		# * 6) last_fit() ----
      		g_model <- extract_spec_parsnip(final_boost_workflow) %>%
       		fit(Y ~ ., data = g_model_data)
		

  			# Propensity score model
			e_model_data$r_idx=factor(e_model_data$r_idx, levels=c(0,1), labels=c("N","Y"))

			common_recipe2 <- recipe(
				r_idx ~ ., data = e_model_data
			)

      		# * 1) model spec rpart ----
			boost_spec <- boost_tree(
	        		tree_depth = tune(),
      	  		trees = tune(),              # add tune() here as well - 20, 50, 100, 200, 300, etc.
        			learn_rate = tune(),
        			mtry = tune(),
	  			min_n = tune(),
      	          	sample_size = 0.8
      		) %>%
			set_mode("classification") %>%
			set_engine("xgboost")

      		# * 2) workflow ----
			boost_workflow <- workflow() %>%
				add_recipe(common_recipe2) %>%
				add_model(boost_spec)

	      	# * 3) resamples ----
			set.seed(123)
			dat_folds <- vfold_cv(e_model_data,v = 10)

	      	# * 4) tune_grid() ----
      		set.seed(123)
      		boost_cube <- grid_latin_hypercube(
        			tree_depth(),
	        		trees(),
      	  		learn_rate(),
        			finalize(mtry(), e_model_data),
        			min_n(),
	        		size = 30
      		)
      		boost_grid <- tune_grid(
        			boost_workflow,
	        		resamples = dat_folds,
      	  		grid = boost_cube,
				control=control_grid(save_pred=T)
	      	)

			final_boost_workflow <- boost_workflow %>%
			finalize_workflow(select_best(boost_grid,metric = "accuracy"))

	      	# * 6) last_fit() ----
      		e_model <- extract_spec_parsnip(final_boost_workflow) %>%
       		fit(r_idx ~ ., data = e_model_data)

		}
		
  		# Predict on test set
		test_data_g=test_data[,-8]
  		g_X_test <- predict(g_model, data.frame(test_data_g))
		test_data_e=test_data[,-1]
  		e_X_test <- predict(e_model, data.frame(test_data_e),type="prob")

  		# Compute doubly robust estimate on test set
		e_X_test$.pred_Y[e_X_test$.pred_Y==0]=0.01
  		mu_estimates[k] <- mean(g_X_test$.pred + (test_data$r_idx / e_X_test$.pred_Y) * (test_data$Y - g_X_test$.pred))

	}

	Y_est_3 <- mean(mu_estimates)
	write.table(data.frame(Y_est_1,Y_est_2,Y_est_3),fno,append=T,row.names=F,col.names=F,sep=",")
}

