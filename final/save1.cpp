#include <stdio.h>
#include <stdlib.h>
#include <limits>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <p4est.h>
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <vector>
#define SIZE 18833843
#define MAX_LAYERS 19 //follows the max_level possible defined by p4est
#define TOTAL_LENGTH std::pow(2,MAX_LAYERS)
#define PT_COUNT 10
//#define DBL_MAX std::numeric_limits<double>::max()
//#define DBL_MIN std::numeric_limits<double>::min()
void get_boundary(long data_size,double* data_ptr, double* max_x, double* max_y, double* max_z,double* min_x, double* min_y, double* min_z);
void discretize(long data_size, double* data_ptr, const double &res, const int &x_offset, const int &y_offset, const int &z_offset, std::vector<long>* pts);

typedef struct
{
  sc_MPI_Comm         mpicomm;
  int                 mpisize;
  int                 mpirank;
}
mpi_context_t;

typedef struct
{
  std::vector<long>* ind_vec_ptr;
  long count;
  long* fused_count;
  long c_x;
  long c_y;
  long c_z;
  long ind;
}user_data_t;





int refine_level = 3;
std::vector<long>* pts_discrete;
std::vector<std::vector<long>*>* v;
std::vector<long>* v_count;
std::vector<long>* all_inds;
std::vector<long>* ind0;
std::vector<long>* ind1;
std::vector<long>* ind2;
std::vector<long>* ind3;
std::vector<long>* ind4;
std::vector<long>* ind5;
std::vector<long>* ind6;
std::vector<long>* ind7;
std::vector<std::vector<long>*>* bins;
std::vector<std::vector<long>*>::iterator data_buf;


long x_bound,y_bound,z_bound;
long lowest_count = SIZE;
//int ind;

static void my_init_fn_first (p8est_t * p4est, p4est_topidx_t which_tree,
         p8est_quadrant_t * quadrant)
{ 
  user_data_t        *data = (user_data_t *) quadrant->p.user_data;

  data->count = SIZE;
  data->c_x = x_bound/2;
  data->c_y = y_bound/2;
  data->c_z = z_bound/2;
  data->ind = 0;
  data->ind_vec_ptr = v->at(0);

  user_data_t        *data2 = (user_data_t *) quadrant->p.user_data;

  std::cerr << quadrant->x << std::endl;
  std::cerr << quadrant->y << std::endl;
  std::cerr << quadrant->z << std::endl;
  std::cerr << "Level" << (int) quadrant->level << std::endl;
  //std::cerr << data2->c_x << " ";

  return;
  //std::vector<long>* data = (std::vector<long>*) quadrant->p.user_data;
}

static void my_init_fn (p8est_t * p4est, p4est_topidx_t which_tree,
         p8est_quadrant_t * quadrant)
{
  user_data_t        *data = (user_data_t *) quadrant->p.user_data;

  long half_side_len = x_bound >> (quadrant->level+1);
  //std::cerr<<half_side_len << std::endl;
  data->count = SIZE;
  data->c_x = quadrant->x+half_side_len;
  data->c_y = quadrant->y+half_side_len;
  data->c_z = quadrant->z+half_side_len;
  data->ind_vec_ptr = *data_buf;

  //std::cerr<< "Size"<<v->size() << std::endl;
  data_buf++;



  std::cerr << quadrant->x << " ";
  std::cerr << quadrant->y << " ";
  std::cerr << quadrant->z << std::endl;
  
  return;
  //std::vector<long>* data = (std::vector<long>*) quadrant->p.user_data;
}


static int
my_refine_fn (p8est_t * p4est, p4est_topidx_t which_tree,
                p8est_quadrant_t * quadrant)
{
  if (quadrant->level >= refine_level){
    return 0;
  }else if (quadrant->level == 0 && false){
    //std::cerr << quadrant->p.which_tree << "original" << std::endl;
    return 1;
  }
  user_data_t        *data = (user_data_t *) quadrant->p.user_data;

  if (data->count < 0.8 * lowest_count){
    if (data->count > PT_COUNT){
      lowest_count = data->count;
    }
    else{
      lowest_count = PT_COUNT;
    }
  }

  //if (data->count > 1.2*lowest_count){
  if (1){
    std::vector<long>* to_split = data->ind_vec_ptr;
    long init_size = data->count/8;
    ind0 = new std::vector<long>();
    ind1 = new std::vector<long>();
    ind2 = new std::vector<long>();
    ind3 = new std::vector<long>();
    ind4 = new std::vector<long>();
    ind5 = new std::vector<long>();
    ind6 = new std::vector<long>();
    ind7 = new std::vector<long>();

    bins = new std::vector<std::vector<long>*>(8);
    *bins = {ind0, ind1, ind2, ind3, ind4, ind5, ind6, ind7};
    
    for(int i=0; i<8; i++){
      bins->at(i)->reserve(init_size);
    }

    for(long i=0; i<to_split->size(); i++){
      int sub_ind = 0;
      if (pts_discrete->at(to_split->at(i)*3) > data->c_x)  sub_ind+=1;
      if (pts_discrete->at(to_split->at(i)*3+1) > data->c_y)  sub_ind+=2;
      if (pts_discrete->at(to_split->at(i)*3+2) > data->c_z)  sub_ind+=4;
      if (sub_ind>7){
        std::cerr <<"inds: " << sub_ind  << " "<< std::endl;
        return 0;
      }

      
      bins->at(sub_ind)->push_back(to_split->at(i));
    }
    std::cerr << std::endl;



    //TO DO free all quadrants
    //v->erase(v->begin()+data->ind);

    std::cerr<< "Size"<<to_split->size() << std::endl;

    
    for(int i=0; i<8; i++){
      //v->insert(v->begin()+(data->ind)+i, bins->at(i));
      v->push_back(bins->at(i));
    }
    data_buf = v->end()-8;
    if (data_buf == v->begin()){
      std::cerr<< "Hitting" << std::endl;
    }
    //delete (data->ind_vec_ptr);
    
    std::cerr << (*(v->begin()))->size() << std::endl;
    //ind = data->ind;
  } 
  //std::cerr<< "v Size"<<v->size() << std::endl;
  std::cerr << "Level, end refine fn" << (int) quadrant->level << std::endl;
  //(quadrant->p.user_data)
  
}


int main(int argc, char** argv)
{
  int                 mpiret;
  int                 wrongusage;
  unsigned            crc;
  const char         *usage;
  mpi_context_t       mpi_context, *mpi = &mpi_context;
  p8est_t            *p8est;
  p8est_connectivity_t *connectivity;
  p8est_refine_t      refine_fn;
  p8est_init_t        init_fn;
  p8est_init_t        init_fn_first;
  p8est_coarsen_t     coarsen_fn;

  refine_fn = my_refine_fn;
  init_fn = my_init_fn;
  init_fn_first = my_init_fn_first;

  /* initialize MPI and p4est internals */
  mpiret = sc_MPI_Init (&argc, &argv);
  SC_CHECK_MPI (mpiret);
  mpi->mpicomm = sc_MPI_COMM_WORLD;
  mpiret = sc_MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
  SC_CHECK_MPI (mpiret);
  mpiret = sc_MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);
  SC_CHECK_MPI (mpiret);

  sc_init (mpi->mpicomm, 1, 1, NULL, SC_LP_DEFAULT);
  p4est_init (NULL, SC_LP_DEFAULT);

  connectivity = p8est_connectivity_new_unitcube ();
  //p8est = p8est_new_ext (mpi->mpicomm, connectivity, 1, 0, 0,
  //                       sizeof (user_data_t), init_fn, NULL);


  /* later changed 'double' to 'int', but that still had issues */
  double* randn = (double *)malloc(SIZE * 3 * sizeof(double));

  //char buff[256];
  FILE *latfile;

  //sprintf(buff,"%s","./test.bin");
  latfile=fopen("test.bin","r");
  fread(randn,sizeof(double),SIZE*3,latfile);
  fclose(latfile);

  
  double min_x,min_y,min_z = DBL_MAX;
  double max_x,max_y,max_z = DBL_MIN;

  get_boundary(SIZE,randn,&max_x,&max_y,&max_z,&min_x,&min_y,&min_z);
  double initial_side_len;
  double res;
  initial_side_len = std::max({max_x-min_x, max_y-min_y, max_z-min_z});
  x_bound = y_bound = z_bound = TOTAL_LENGTH;
  res =  initial_side_len / x_bound;
  std::cerr << max_x << std::endl;
  std::cerr << min_x << std::endl;
  std::cerr << max_y << std::endl;
  std::cerr << min_y << std::endl;
  std::cerr << max_z << std::endl;
  std::cerr << min_z << std::endl;
  std::cerr << initial_side_len << std::endl;
  std::cerr << res << std::endl;
  std::cerr << x_bound << std::endl;

  int x_offset, y_offset, z_offset;
  x_offset = -floor(min_x/res);
  y_offset = -floor(min_y/res);
  z_offset = -floor(min_z/res);
  std::cerr << x_offset << std::endl;
  std::cerr << y_offset << std::endl;
  std::cerr << z_offset << std::endl;

  

  v = new std::vector<std::vector<long>*>();
  v->reserve(8);
  pts_discrete = new std::vector<long>(SIZE * 3);
  all_inds = new std::vector<long>(SIZE);
  std::generate(all_inds->begin(), all_inds->end(), [n = 0]() mutable { return n++ ; });

  discretize(SIZE, randn, res, x_offset, y_offset, z_offset, pts_discrete);

  v->push_back(all_inds);
  //ind = 0;

/*  
*  debug
*  for(int i=500000; i< 500099; i++){
*      std::cerr<<x_discrete[i] << std::endl;
*      std::cerr<<y_discrete[i] << std::endl;
*      std::cerr<<z_discrete[i] << std::endl;
*  }
*/

  p8est = p8est_new(mpi->mpicomm, connectivity,sizeof(user_data_t), init_fn_first, NULL);
  p4est_gloidx_t num_quadrants = p8est->global_num_quadrants;
  //To do a breadth-first expanding, avoid using the built-in recursive approach
  do{

    num_quadrants = p8est->global_num_quadrants;
    p8est_refine (p8est, 0, refine_fn, init_fn);

    long w = 0;
    for (auto it = v->begin();it<v->end();it++){

      
      w += (*it)->size();
    }
    std::cerr << "Period size ";
    std::cerr << w << std::endl;
  }while (num_quadrants != p8est->global_num_quadrants);
  

  free(randn);
  delete pts_discrete;
  delete v;
  delete all_inds;

  p8est_destroy (p8est);

  p8est_connectivity_destroy (connectivity);

  /* clean up and exit */
  sc_finalize ();

  mpiret = sc_MPI_Finalize ();
  SC_CHECK_MPI (mpiret);

  for (auto it=v->begin();it<v->end();it++){
    if (*it != NULL){
      delete (*it);
    }
  }

  return 0;
}



void get_boundary(long data_size,double* data_ptr, double* max_x, double* max_y, double* max_z,double* min_x, double* min_y, double* min_z){
    for(long i=0;i<data_size*3;i+=3){
        if (data_ptr[i] > *max_x){
            *max_x = data_ptr[i];
        }

        if (data_ptr[i+1] > *max_y){
            *max_y = data_ptr[i+1];
        }

        if (data_ptr[i+2] > *max_z){
            *max_z = data_ptr[i+2];
        }

        if (data_ptr[i] < *min_x){
            *min_x = data_ptr[i];
        }

        if (data_ptr[i+1] < *min_y){
            *min_y = data_ptr[i+1];
        }

        if (data_ptr[i+2] < *min_z){
            *min_z = data_ptr[i+2];
        }
    }
}

void discretize(long data_size, double* data_ptr, const double &res, const int &x_offset, const int &y_offset, const int &z_offset, std::vector<long>* pts){
    double res_recp = 1.0/res;
    for(long j=0;j<data_size;j+=3){
        (*pts)[j] = floor(data_ptr[j] * res_recp + x_offset);   //fma
        (*pts)[j+1] = floor(data_ptr[j+1] * res_recp + y_offset);
        (*pts)[j+2] = floor(data_ptr[j+2] * res_recp + z_offset);
    }
}

