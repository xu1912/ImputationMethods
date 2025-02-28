expit=function(x){
        return(exp(x)/(1+exp(x)))
}


ns=100
N=500
m_rate=0.5
K=5
mrs=c()
model="linear"
#model="nonlinear"
#model="highnonlinear"
for(i in 1:ns){

	x1=rnorm(N, mean=0, sd=1)
	x2=rchisq(N, df=5)
	x3=rbinom(N, size=1, prob=0.4)
	x4=rmultinom(N, size=1, prob=c(0.2,0.3,0.5))
	x41=t(x4)[,2]
	x42=t(x4)[,3]
	err=rnorm(N, mean=0, sd=1)

	if(model=="linear"){
		Y = 1 + 0.6*x1 - 0.3*x2 + 1.1*x3 + 0.65*x41 + 2.47*x42 + err
		mn="A"
		ccc=-2.25
		misng_rate_v=ccc+0.1*x1+0.3*x2+0.6*x3+0.5*x41+0.9*x42
		misng_rate=exp(misng_rate_v)/(1+exp(misng_rate_v))
	}

	if(model=="nonlinear"){
		Y = 1 - 0.25*x1^3 + 0.07*x2^(2) - 0.11*x1*x2 + 0.68*x1*x2*x3*x41 - 0.97*x3*x42 + err
		mn="B"
		ccc=6.65
		misng_rate_v=ccc+0.15*x1^3-0.35*x2^2+x3+0.15*x1*x2+0.8*x2*x41*x42
		misng_rate=exp(misng_rate_v)/(1+exp(misng_rate_v))
	}

	if(model=="highnonlinear"){
		Y = 1 + abs(x1)^(1/3)*x2^(1/5)*log(abs(x1+x3)) + sin(x1^2 + 1.5*x3 - 0.2*x41) + err
		mn="C"
		ccc=-2.17
		misng_rate_v=ccc+cos(x1^3*x2^2*x3)+log(abs(x1*x2))+x2^(1/3)*x41+x3*x42
		misng_rate=exp(misng_rate_v)/(1+exp(misng_rate_v))
	}

	r_idx=rbinom(N, 1, misng_rate)
	mrs[i]=mean(r_idx)
	mrs[i]=mean(Y)

	r_idx_tp=as.logical(r_idx)
	X=as.matrix(cbind(x1,x2,x3,x41,x42))

	wres=data.frame(cbind(Y,X,r_idx))
	fn=paste("J:/Book/m", mn, "/",i,".csv",sep="")
	write.csv(wres, fn,row.names=F)
}

summary(mrs)
mean(mrs)
