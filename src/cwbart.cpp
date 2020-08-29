/*
 *  BART: Bayesian Additive Regression Trees
 *  Copyright (C) 2017 Robert McCulloch and Rodney Sparapani
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, a copy is available at
 *  https://www.R-project.org/Licenses/GPL-2
 */

#include <RcppArmadillo.h>
#include "tree.h"
#include "Rmath.h"
#include "treefuns.h"
#include "info.h"
#include "bartfuns.h"
#include "bd.h"
#include "bart.h"
#include "heterbart.h"
#include "my_helper.h"

#ifndef NoRcpp

#define TRDRAW(a, b) trdraw(a, b)
#define TEDRAW(a, b) tedraw(a, b)

RcppExport SEXP cwbart(
   SEXP _in,            //number of observations in training data
   SEXP _ip,		//dimension of x
   SEXP _inp,		//number of observations in test data
   SEXP _ix,		//x, train,  pxn (transposed so rows are contiguous in memory)
   SEXP _iy,		//y, train,  nx1
   SEXP _ixp,		//x, test, pxnp (transposed so rows are contiguous in memory)
   SEXP _im,		//number of trees
   SEXP _inc,		//number of cut points
   SEXP _ind,		//number of kept draws (except for thinnning ..)
   SEXP _iburn,		//number of burn-in draws skipped
   SEXP _ipower,
   SEXP _ibase,
   SEXP _itau,
   SEXP _inu,
   SEXP _ilambda,
   SEXP _isigest,
   SEXP _iw,
   SEXP _idart,
   SEXP _itheta,
   SEXP _iomega,
   SEXP _igrp,
   SEXP _ia,
   SEXP _ib,
   SEXP _irho,
   SEXP _iaug,
   SEXP _inkeeptrain,
   SEXP _inkeeptest,
   SEXP _inkeeptestme,
   SEXP _inkeeptreedraws,
   SEXP _inprintevery,
   SEXP _iz,
   SEXP _idistance,

   SEXP _idistance_all,
   SEXP _inew_loc,

//   SEXP _treesaslists,
   SEXP _Xinfo,
   SEXP _range_select_sd,
   SEXP _smoothness_select_sd,
   SEXP _sigma2_prior_a,
   SEXP _sigma2_prior_b,
   SEXP _tau2_prior_a,
   SEXP _tau2_prior_b,
   SEXP _range,
   SEXP _smoothness,
   SEXP _tau2,
   SEXP _range_prior_mean,
   SEXP _range_prior_sd,
   SEXP _smoothness_prior_mean,
   SEXP _smoothness_prior_sd

)
{

   //--------------------------------------------------
   //process args
   Rcpp::NumericMatrix z(_iz);
   int nrow = z.nrow(), k = z.ncol();
   arma::mat zz(z.begin(), nrow, k, false);

   Rcpp::NumericMatrix distance(_idistance);
   nrow = distance.nrow();
   k = distance.ncol();
   arma::mat ddistance(distance.begin(), nrow, k, false);
   int nn = nrow;

   Rcpp::NumericMatrix distance_all(_idistance_all);
   nrow = distance_all.nrow();
   k = distance_all.ncol();
   arma::mat ddistance_all(distance_all.begin(), nrow, k, false);
   int nn2 = nrow - nn;

   bool new_loc = Rcpp::as<bool>(_inew_loc);

   double range_sd = Rcpp::as<double>(_range_select_sd);
   double smoothness_sd = Rcpp::as<double>(_smoothness_select_sd);

   double sigma2_prior_a = Rcpp::as<double>(_sigma2_prior_a);
   double sigma2_prior_b = Rcpp::as<double>(_sigma2_prior_b);
   double tau2_prior_a = Rcpp::as<double>(_tau2_prior_a);
   double tau2_prior_b = Rcpp::as<double>(_tau2_prior_b);
   double range = Rcpp::as<double>(_range);
   double smoothness = Rcpp::as<double>(_smoothness);
   double tau2 = Rcpp::as<double>(_tau2);
   double range_prior_mean = Rcpp::as<double>(_range_prior_mean);
   double range_prior_sd = Rcpp::as<double>(_range_prior_sd);
   double smoothness_prior_mean = Rcpp::as<double>(_smoothness_prior_mean);
   double smoothness_prior_sd = Rcpp::as<double>(_smoothness_prior_sd);


   size_t n = Rcpp::as<int>(_in);
   size_t p = Rcpp::as<int>(_ip);
   size_t np = Rcpp::as<int>(_inp);
   Rcpp::NumericVector  xv(_ix);
   double *ix = &xv[0];
   Rcpp::NumericVector  yv(_iy);
   double *iy = &yv[0];
   Rcpp::NumericVector  xpv(_ixp);
   double *ixp = &xpv[0];
   size_t m = Rcpp::as<int>(_im);
   Rcpp::IntegerVector _nc(_inc);
   int *numcut = &_nc[0];
   //size_t nc = Rcpp::as<int>(_inc);
   size_t nd = Rcpp::as<int>(_ind);
   size_t burn = Rcpp::as<int>(_iburn);
   double mybeta = Rcpp::as<double>(_ipower);
   double alpha = Rcpp::as<double>(_ibase);
   double tau = Rcpp::as<double>(_itau);
   double nu = Rcpp::as<double>(_inu);
   double lambda = Rcpp::as<double>(_ilambda);
   double sigma=Rcpp::as<double>(_isigest);
   double sigma2 = sigma*sigma;
   Rcpp::NumericVector  wv(_iw);
   double *iw = &wv[0];
   bool dart;
   if(Rcpp::as<int>(_idart)==1) dart=true;
   else dart=false;
   double a = Rcpp::as<double>(_ia);
   double b = Rcpp::as<double>(_ib);
   double rho = Rcpp::as<double>(_irho);
   bool aug;
   if(Rcpp::as<int>(_iaug)==1) aug=true;
   else aug=false;
   double theta = Rcpp::as<double>(_itheta);
   double omega = Rcpp::as<double>(_iomega);
   Rcpp::IntegerVector _grp(_igrp);
   int *grp = &_grp[0];
   size_t nkeeptrain = Rcpp::as<int>(_inkeeptrain);
   size_t nkeeptest = Rcpp::as<int>(_inkeeptest);
   size_t nkeeptestme = Rcpp::as<int>(_inkeeptestme);
   size_t nkeeptreedraws = Rcpp::as<int>(_inkeeptreedraws);
   size_t printevery = Rcpp::as<int>(_inprintevery);
//   int treesaslists = Rcpp::as<int>(_treesaslists);
   Rcpp::NumericMatrix Xinfo(_Xinfo);
//   Rcpp::IntegerMatrix varcount(nkeeptreedraws, p);

   //return data structures (using Rcpp)
   Rcpp::NumericVector trmean(n); //train
   Rcpp::NumericVector temean(np);
   Rcpp::NumericVector sdraw(nd+burn);
   Rcpp::NumericMatrix trdraw(nkeeptrain,n);
   Rcpp::NumericMatrix tedraw(nkeeptest,np);
//   Rcpp::List list_of_lists(nkeeptreedraws*treesaslists);
   Rcpp::NumericMatrix varprb(nkeeptreedraws,p);
   Rcpp::IntegerMatrix varcnt(nkeeptreedraws,p);

   //random number generation
   arn gen;

   heterbart bm(m);

   if(Xinfo.size()>0) {
     xinfo _xi;
     _xi.resize(p);
     for(size_t i=0;i<p;i++) {
       _xi[i].resize(numcut[i]);
       //Rcpp::IntegerVector cutpts(Xinfo[i]);
       for(size_t j=0;j<numcut[i];j++) _xi[i][j]=Xinfo(i, j);
     }
     bm.setxinfo(_xi);
   }
#else

#define TRDRAW(a, b) trdraw[a][b]
#define TEDRAW(a, b) tedraw[a][b]

void cwbart(
   size_t n,            //number of observations in training data
   size_t p,		//dimension of x
   size_t np,		//number of observations in test data
   double* ix,		//x, train,  pxn (transposed so rows are contiguous in memory)
   double* iy,		//y, train,  nx1
   double* ixp,		//x, test, pxnp (transposed so rows are contiguous in memory)
   size_t m,		//number of trees
   int* numcut,		//number of cut points
   size_t nd,		//number of kept draws (except for thinnning ..)
   size_t burn,		//number of burn-in draws skipped
   double mybeta,
   double alpha,
   double tau,
   double nu,
   double lambda,
   double sigma,
   double* iw,
   bool dart,
   double theta,
   double omega,
   int *grp,
   double a,
   double b,
   double rho,
   bool aug,
   size_t nkeeptrain,
   size_t nkeeptest,
   size_t nkeeptestme,
   size_t nkeeptreedraws,
   size_t printevery,
   // double* z,
   // double* location,
   // double* distance,
   // size_t nn,
//   int treesaslists,
   unsigned int n1, // additional parameters needed to call from C++
   unsigned int n2,
   double* trmean,
   double* temean,
   double* sdraw,
   double* _trdraw,
   double* _tedraw
)
{

   //return data structures (using C++)
   std::vector<double*> trdraw(nkeeptrain);
   std::vector<double*> tedraw(nkeeptest);

   for(size_t i=0; i<nkeeptrain; ++i) trdraw[i]=&_trdraw[i*n];
   for(size_t i=0; i<nkeeptest; ++i) tedraw[i]=&_tedraw[i*np];

   std::vector< std::vector<size_t> > varcnt;
   std::vector< std::vector<double> > varprb;

   //random number generation
   arn gen(n1, n2);

   heterbart bm(m);
#endif

   for(size_t i=0;i<n;i++) trmean[i]=0.0;
   for(size_t i=0;i<np;i++) temean[i]=0.0;

   printf("*****Into main of wbart\n");
   //-----------------------------------------------------------

   size_t skiptr,skipte,skipteme,skiptreedraws;
   if(nkeeptrain) {skiptr=nd/nkeeptrain;}
   else skiptr = nd+1;
   if(nkeeptest) {skipte=nd/nkeeptest;}
   else skipte=nd+1;
   if(nkeeptestme) {skipteme=nd/nkeeptestme;}
   else skipteme=nd+1;
   if(nkeeptreedraws) {skiptreedraws = nd/nkeeptreedraws;}
   else skiptreedraws=nd+1;

   //--------------------------------------------------
   //print args
   printf("*****Data:\n");
   printf("data:n,p,np: %zu, %zu, %zu\n",n,p,np);
   printf("y1,yn: %lf, %lf\n",iy[0],iy[n-1]);
   printf("x1,x[n*p]: %lf, %lf\n",ix[0],ix[n*p-1]);
   if(np) printf("xp1,xp[np*p]: %lf, %lf\n",ixp[0],ixp[np*p-1]);
   printf("*****Number of Trees: %zu\n",m);
   printf("*****Number of Cut Points: %d ... %d\n", numcut[0], numcut[p-1]);
   printf("*****burn and ndpost: %zu, %zu\n",burn,nd);
   printf("*****Prior:beta,alpha,tau,nu,lambda: %lf,%lf,%lf,%lf,%lf\n",
                   mybeta,alpha,tau,nu,lambda);
   printf("*****sigma: %lf\n",sigma);
   printf("*****w (weights): %lf ... %lf\n",iw[0],iw[n-1]);
   cout << "*****Dirichlet:sparse,theta,omega,a,b,rho,augment: "
	<< dart << ',' << theta << ',' << omega << ',' << a << ','
	<< b << ',' << rho << ',' << aug << endl;
   printf("*****nkeeptrain,nkeeptest,nkeeptestme,nkeeptreedraws: %zu,%zu,%zu,%zu\n",
               nkeeptrain,nkeeptest,nkeeptestme,nkeeptreedraws);
   printf("*****printevery: %zu\n",printevery);
   printf("*****skiptr,skipte,skipteme,skiptreedraws: %zu,%zu,%zu,%zu\n",skiptr,skipte,skipteme,skiptreedraws);

   //--------------------------------------------------
   //heterbart bm(m);
   bm.setprior(alpha,mybeta,tau);
   bm.setdata(p,n,ix,iy,numcut);
   bm.setdart(a,b,rho,aug,dart,theta,omega);

   //--------------------------------------------------
   //sigma
   //gen.set_df(n+nu);
   double *svec = new double[n];
   for(size_t i=0;i<n;i++) svec[i]=iw[i]*sigma;

   //--------------------------------------------------

   std::stringstream treess;  //string stream to write trees to
   treess.precision(10);
   treess << nkeeptreedraws << " " << m << " " << p << endl;
   // dart iterations
   std::vector<double> ivarprb (p,0.);
   std::vector<size_t> ivarcnt (p,0);

   //--------------------------------------------------
   //temporary storage
   //out of sample fit
   double* fhattest=0; //posterior mean for prediction
   if(np) { fhattest = new double[np]; }
   double restemp=0.0,rss=0.0;

   // https://stackoverflow.com/questions/14253069/convert-rcpparmadillo-vector-to-rcpp-vector

   double *new_y = new double[n];
   arma::vec y_corr(n);
   for(size_t i=0;i<n;i++){y_corr(i)=0.0;};
   arma::vec zero_nn(nn);
   for(size_t i=0;i<nn;i++){zero_nn(i)=0.0;};

   arma::vec tau2_record(nd+burn);
   arma::vec range_record(nd+burn);
   arma::vec smoothness_record(nd+burn);
   arma::mat w;
   arma::mat w_record(nd+burn,nn);
   arma::mat w_record_test(nd+burn,nn2);
   arma::mat vvv;
   arma::mat xxx;
   arma::mat siginv;
   arma::mat S(nn2+nn,nn2+nn);
   arma::mat S11(nn2,nn2);
   arma::mat S12(nn2,nn);
   arma::mat S22inv(nn,nn);
   arma::mat VVV2(nn2,nn2);
   arma::vec MMM(nn2);
   arma::vec temp_vec(nn2);
   arma::vec yy(n);
   arma::mat vvv_record;
   arma::mat VVV2_record;
   double ssr_yy;
   double curll;
   double canll;
   arma::mat temp;

   double can_range;
   double can_smoothness;

   tau2_record(0) = tau2;
   range_record(0) = range;
   smoothness_record(0) = smoothness;

   double mh;
   double temp1;
   double temp2;
   double temp_r;
   double dett;
   Rcpp::NumericVector tempvec1[1];
   Rcpp::NumericVector tempvec2[1];
   arma::mat tempmat1;


   // arma::mat sigma_matrix;
   arma::vec covparms(3);
   arma::vec can_covparms(3);
   covparms(0) = tau2;
   covparms(1) = range;
   covparms(2) = smoothness;
   arma::mat sigma_matrix = MaternFun(ddistance, covparms);
   arma::mat can_sigma_matrix;



   //--------------------------------------------------
   //mcmc
   printf("\nMCMC\n");
   //size_t index;
   size_t trcnt=0; //count kept train draws
   size_t tecnt=0; //count kept test draws
   size_t temecnt=0; //count test draws into posterior mean
   size_t treedrawscnt=0; //count kept bart draws
   bool keeptest,keeptestme,keeptreedraw;

   time_t tp;
   int time1 = time(&tp);
   xinfo& xi = bm.getxinfo();

   for(size_t i=0;i<(nd+burn);i++) {
      if(i%printevery==0) {printf("done %zu (out of %lu)\n",i,nd+burn);}
          //vvv_record = arma::join_rows(vvv_record,vvv);}

      //cout << "iy[0] " << iy[0] << " y_corr[0] " << y_corr(0) << " ";
      //cout << "iy[1] " << iy[1] << " y_corr[1] " << y_corr(1) << " ";
      for(size_t ii=0;ii<n;ii++){new_y[ii] = iy[ii] - y_corr(ii);}
      bm.resety(new_y);
      //cout << " new_y[0] " << bm.y[0] << " " ;
      //cout << " new_y[1] " << bm.y[1] << " ";

      if(i==(burn/2)&&dart) bm.startdart();
      //draw bart
      bm.draw(svec,gen);

      for(size_t ii=0; ii<n;ii++) {yy(ii)=(iy[ii]-bm.f(ii));}
      siginv = arma::inv(sigma_matrix / tau2);
      vvv = arma::inv(zz.t() * zz / sigma2 +  siginv / tau2);
      xxx = zz.t() * yy;

      //cout << endl;
      //cout << "sigma2 " << sigma2 << " " << "tau2 " << tau2 << " " << "range " << range << " " << "smoothness " << smoothness << " ";
      //cout << "training " ;
      w = mvrnormArma(1, vvv * xxx / sigma2, vvv);
      w_record.row(i) = w.row(0);
      y_corr = zz * w.t();
      ssr_yy=0.0;
      for(size_t ii=0; ii<n;ii++) ssr_yy += (yy(ii)-y_corr(ii))*(yy(ii)-y_corr(ii));
      sigma2 = 1/R::rgamma(sigma2_prior_a + n/2, 1/(ssr_yy/2 + sigma2_prior_b));
//       sigma = sqrt(sigma2);
//       sdraw[i] = sigma;

      // update tau2
      temp = w*siginv*w.t();  // w here is a row vec
      tau2 = 1/R::rgamma(tau2_prior_a + nn/2, 1/(temp(0,0)/2 + tau2_prior_b));
      //cout << "nn " << nn << " " << "temp " <<temp(0,0)/2 << " ";
      //cout << "tau2 " << tau2 << " ";
      tau2_record(i) = tau2;
      covparms(0) = tau2;
      sigma_matrix = MaternFun(ddistance, covparms);
      curll = dmvnrm_arma(w, zero_nn.t(), sigma_matrix)(0);
      can_covparms = covparms;
      can_range = R::rnorm(range,range_sd);
      can_covparms(1) = can_range;
      can_sigma_matrix = MaternFun(ddistance, can_covparms);
      canll = dmvnrm_arma(w, zero_nn.t(), can_sigma_matrix)(0);
      temp1 = R::dnorm(can_range,range_prior_mean,range_prior_sd,1);
      temp2 = R::dnorm(range,range_prior_mean,range_prior_sd,1);
      mh = canll-curll+temp1-temp2;
      temp_r = log(R::runif(0,1));
//       if(temp_r < mh){
//         range = can_range;
//         sigma_matrix = can_sigma_matrix;
//         curll = canll;
//         covparms(1) = can_range;
//       }

      // https://kevinushey.github.io/blog/2015/04/05/debugging-with-valgrind/
      // smoothness
      sigma_matrix = MaternFun(ddistance, covparms);
      can_covparms = covparms;
      can_smoothness = R::rnorm(smoothness,smoothness_sd);
      can_covparms(2) = can_smoothness;
      can_sigma_matrix = MaternFun(ddistance, can_covparms);
      canll = dmvnrm_arma(w, zero_nn.t(), can_sigma_matrix)(0);
      temp1 = R::dnorm(can_smoothness,smoothness_prior_mean,smoothness_prior_sd,1);
      temp2 = R::dnorm(smoothness,smoothness_prior_mean,smoothness_prior_sd,1);
      mh = canll-curll+temp1-temp2;
      // https://zenglix.github.io/Rcpp_basic/
      temp_r = log(R::runif(0,1));
//       if(temp_r < mh){
//         smoothness = can_smoothness;
//         sigma_matrix = can_sigma_matrix;
//         curll = canll;
//         covparms(2) = can_smoothness;
//       }

      range_record(i) = range;
      smoothness_record(i) = smoothness;

      if(i>=burn) {

         if (new_loc){
            // https://en.wikipedia.org/wiki/Multivariate_normal_distribution ;
            // https://www4.stat.ncsu.edu/~reich/SpatialStats/code/Bayes_Krig.html ;
            // http://arma.sourceforge.net/docs.html#submat ;
            S = MaternFun(ddistance_all, covparms);
            S11 = S.submat(0,0,nn2-1,nn2-1);
            S12 = S.submat(0,nn2,nn2-1,nn2+nn-1);
            S22inv = arma::inv(S.submat(nn2,nn2,nn2+nn-1,nn2+nn-1));
            VVV2 = S11 - S12 * S22inv * (S12.t());
            MMM = S12 * S22inv * w.t();
            //cout << "testing ";
            w_record_test.row(i) = mvrnormArma(1, MMM, VVV2).row(0);
         }

         for(size_t k=0;k<n;k++) trmean[k]+=bm.f(k);
         if(nkeeptrain && (((i-burn+1) % skiptr) ==0)) {
            //index = trcnt*n;;
            //for(size_t k=0;k<n;k++) trdraw[index+k]=bm.f(k);
            for(size_t k=0;k<n;k++) TRDRAW(trcnt,k)=bm.f(k);
            trcnt+=1;
         }
         keeptest = nkeeptest && (((i-burn+1) % skipte) ==0) && np;
         keeptestme = nkeeptestme && (((i-burn+1) % skipteme) ==0) && np;
         if(keeptest || keeptestme) bm.predict(p,np,ixp,fhattest);
         if(keeptest) {
            //index=tecnt*np;
            //for(size_t k=0;k<np;k++) tedraw[index+k]=fhattest[k];
            for(size_t k=0;k<np;k++) TEDRAW(tecnt,k)=fhattest[k];
            tecnt+=1;
         }
         if(keeptestme) {
            for(size_t k=0;k<np;k++) temean[k]+=fhattest[k];
            temecnt+=1;
         }
         keeptreedraw = nkeeptreedraws && (((i-burn+1) % skiptreedraws) ==0);
         if(keeptreedraw) {
//	   #ifndef NoRcpp
//	   Rcpp::List lists(m*treesaslists);
//	   #endif

            for(size_t j=0;j<m;j++) {
	      treess << bm.gettree(j);
/*
	      #ifndef NoRcpp
	      varcount.row(treedrawscnt)=varcount.row(treedrawscnt)+bm.gettree(j).tree2count(p);
	      if(treesaslists) lists(j)=bm.gettree(j).tree2list(xi, 0., 1.);
	      #endif
*/
	    }
            #ifndef NoRcpp
//	    if(treesaslists) list_of_lists(treedrawscnt)=lists;
	    ivarcnt=bm.getnv();
	    ivarprb=bm.getpv();
	    size_t k=(i-burn)/skiptreedraws;
	    for(size_t j=0;j<p;j++){
	      varcnt(k,j)=ivarcnt[j];
	      //varcnt(i-burn,j)=ivarcnt[j];
	      varprb(k,j)=ivarprb[j];
	      //varprb(i-burn,j)=ivarprb[j];
	    }
            #else
	    varcnt.push_back(bm.getnv());
	    varprb.push_back(bm.getpv());
	    #endif

            treedrawscnt +=1;
         }
      }
   }
   int time2 = time(&tp);
   printf("time: %ds\n",time2-time1);
   for(size_t k=0;k<n;k++) trmean[k]/=nd;
   for(size_t k=0;k<np;k++) temean[k]/=temecnt;
   printf("check counts\n");
   printf("trcnt,tecnt,temecnt,treedrawscnt: %zu,%zu,%zu,%zu\n",trcnt,tecnt,temecnt,treedrawscnt);
   //--------------------------------------------------
   //PutRNGstate();

   if(fhattest) delete[] fhattest;
   if(svec) delete [] svec;

   // Rcpp::NumericMatrix Wret = Rcpp::NumericVector(wrap(zz));
   //--------------------------------------------------
   //return
#ifndef NoRcpp
   Rcpp::List ret;
   ret["sigma"]=sdraw;
   ret["yhat.train.mean"]=trmean;
   ret["yhat.train"]=trdraw;
   ret["yhat.test.mean"]=temean;
   ret["yhat.test"]=tedraw;
   //ret["varcount"]=varcount;
   ret["varcount"]=varcnt;
   ret["varprob"]=varprb;
   ret["tau2"]= Rcpp::NumericVector(wrap(tau2_record));
   ret["range"]= Rcpp::NumericVector(wrap(range_record));
   ret["smoothness"]= Rcpp::NumericVector(wrap(smoothness_record));
   ret["w"]= Rcpp::NumericMatrix(wrap(w_record));
   ret["w_test"]= Rcpp::NumericMatrix(wrap(w_record_test));
   // ret["zz"]= zz;

   //for(size_t i=0;i<m;i++) {
    //  bm.gettree(i).pr();
   //}

   Rcpp::List xiret(xi.size());
   for(size_t i=0;i<xi.size();i++) {
      Rcpp::NumericVector vtemp(xi[i].size());
      std::copy(xi[i].begin(),xi[i].end(),vtemp.begin());
      xiret[i] = Rcpp::NumericVector(vtemp);
   }

   Rcpp::List treesL;
   //treesL["nkeeptreedraws"] = Rcpp::wrap<int>(nkeeptreedraws); //in trees
   //treesL["ntree"] = Rcpp::wrap<int>(m); //in trees
   //treesL["numx"] = Rcpp::wrap<int>(p); //in cutpoints
   treesL["cutpoints"] = xiret;
   treesL["trees"]=Rcpp::CharacterVector(treess.str());
//   if(treesaslists) treesL["lists"]=list_of_lists;
   ret["treedraws"] = treesL;

   return ret;
#else

#endif

}
