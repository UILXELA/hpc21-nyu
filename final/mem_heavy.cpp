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
#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define SIZE 18833843
#define MAX_LAYERS 19 //follows the max_level possible defined by p4est
#define TOTAL_LENGTH std::pow(2,MAX_LAYERS)
#define PT_COUNT 10
//#define DBL_MAX std::numeric_limits<double>::max()
//#define DBL_MIN std::numeric_limits<double>::min()
void get_boundary(long data_size,double* data_ptr, double& max_x, double& max_y, double& max_z,double& min_x, double& min_y, double& min_z);
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


int refine_level = 18;
std::vector<long>* pts_discrete;
std::vector<std::vector<long>*>* v;
std::vector<long>* v_count;
std::vector<long>* all_inds;
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
  
  data->c_x = quadrant->x+half_side_len;
  data->c_y = quadrant->y+half_side_len;
  data->c_z = quadrant->z+half_side_len;
  data->ind_vec_ptr = *data_buf;
  data->count = data->ind_vec_ptr->size();

  //std::cerr<< "Size "<<data->count << std::endl;
  data_buf++;



  //std::cerr << quadrant->x << " ";
  //std::cerr << quadrant->y << " ";
  //std::cerr << quadrant->z << std::endl;
  
  return;
  //std::vector<long>* data = (std::vector<long>*) quadrant->p.user_data;
}


static int
my_refine_fn (p8est_t * p4est, p4est_topidx_t which_tree,
                p8est_quadrant_t * quadrant)
{
  if (quadrant->level >= refine_level){
    return 0;
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

  if (data->count > 1.2*lowest_count || (quadrant->level == 0 )){
  //if (1){
    std::vector<long>* to_split = data->ind_vec_ptr;
    long init_size = data->count/8;
    std::vector<long>* ind0 = new std::vector<long>();
    std::vector<long>* ind1 = new std::vector<long>();
    std::vector<long>* ind2 = new std::vector<long>();
    std::vector<long>* ind3 = new std::vector<long>();
    std::vector<long>* ind4 = new std::vector<long>();
    std::vector<long>* ind5 = new std::vector<long>();
    std::vector<long>* ind6 = new std::vector<long>();
    std::vector<long>* ind7 = new std::vector<long>();

    std::vector<std::vector<long>*> bins = {ind0, ind1, ind2, ind3, ind4, ind5, ind6, ind7};
    
    //for(int i=0; i<8; i++){
    //  bins[i]->reserve(data->count);
      //bins[i]->reserve(SIZE);
    //}

    #pragma omp parallel
    {
    std::vector<long> ind0p, ind1p, ind2p, ind3p, ind4p, ind5p, ind6p, ind7p;
    std::vector<std::vector<long>> bins_p = {ind0p, ind1p, ind2p, ind3p, ind4p, ind5p, ind6p, ind7p}; 
    for(int i=0; i<8; i++){
      bins_p[i].reserve(data->count);
      //bins[i]->reserve(SIZE);
    } 
      #pragma omp for nowait schedule(static)
      for(long i=0; i<to_split->size(); i++){
        //std::cerr << i  << std::endl;
        int sub_ind = 0;
        if (pts_discrete->at(to_split->at(i)*3) > data->c_x)  sub_ind+=1;
        if (pts_discrete->at(to_split->at(i)*3+1) > data->c_y)  sub_ind+=2;
        if (pts_discrete->at(to_split->at(i)*3+2) > data->c_z)  sub_ind+=4;
        //if (sub_ind>7){
        //  std::cerr <<"inds: " << sub_ind  << " "<< std::endl;
        //  return 0;
        //}
        
        bins_p[sub_ind].push_back(to_split->at(i));
        
      }
      #pragma omp critical
      for(int i=0; i<8; i++){
        //bins[i]->reserve(init_size);
        bins[i]->insert(bins[i]->end(),bins_p[i].begin(), bins_p[i].end());
      }
    };

    /*#pragma omp parallel for schedule(static)
    for(long i=0; i<to_split->size(); i++){
      //std::cerr << i  << std::endl;
      int sub_ind = 0;
      if (pts_discrete->at(to_split->at(i)*3) > data->c_x)  sub_ind+=1;
      if (pts_discrete->at(to_split->at(i)*3+1) > data->c_y)  sub_ind+=2;
      if (pts_discrete->at(to_split->at(i)*3+2) > data->c_z)  sub_ind+=4;
      //if (sub_ind>7){
      //  std::cerr <<"inds: " << sub_ind  << " "<< std::endl;
      //  return 0;
      //}
      
      bins[sub_ind]->push_back(to_split->at(i));
    }
    */


    //TO DO free all quadrants
    //v->erase(v->begin()+data->ind);

    //std::cerr<< "Size "<<to_split->size() << std::endl;

    //sequential required
    for(int i=0; i<8; i++){
      //v->insert(v->begin()+(data->ind)+i, bins->at(i));
      v->push_back(bins[i]);
    }
    data_buf = v->end()-8;
    //if (data_buf == v->begin()){
    //  std::cerr<< "Hitting" << std::endl;
    //}
    data->ind_vec_ptr->clear();
    //delete (data->ind_vec_ptr);

    return 1;
    
    //std::cerr << (*(v->begin()))->size() << std::endl;
    //ind = data->ind;
  }else{
    return 0;
  }
  //std::cerr<< "v Size"<<v->size() << std::endl;
  //std::cerr << "Level, end refine fn" << (int) quadrant->level << std::endl;
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
  int rank, size;

  MPI_Status status;

  refine_fn = my_refine_fn;
  init_fn = my_init_fn;
  init_fn_first = my_init_fn_first;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int num_threads = 1;
	#ifdef _OPENMP
		num_threads = omp_get_max_threads();
	#endif

	printf("\nRunning with omp threads: %d\n", num_threads);

  /* initialize MPI and p4est internals */
  //mpiret = sc_MPI_Init (&argc, &argv);
  //SC_CHECK_MPI (mpiret);
  mpi->mpicomm = sc_MPI_COMM_WORLD;
  //mpiret = sc_MPI_Comm_size (mpi->mpicomm, &mpi->mpisize);
  //SC_CHECK_MPI (mpiret);
  //mpiret = sc_MPI_Comm_rank (mpi->mpicomm, &mpi->mpirank);
  //SC_CHECK_MPI (mpiret);

  //sc_init (mpi->mpicomm, 1, 1, NULL, SC_LP_DEFAULT);
  MPI_Barrier(MPI_COMM_WORLD);
  p4est_init (NULL, SC_LP_DEFAULT);

  connectivity = p8est_connectivity_new_unitcube ();
  //p8est = p8est_new_ext (mpi->mpicomm, connectivity, 1, 0, 0,
  //                       sizeof (user_data_t), init_fn, NULL);


  /* later changed 'double' to 'int', but that still had issues */
  double* randn = (double *)malloc(SIZE * 3 * sizeof(double));

  //char buff[256];
  //FILE *latfile;

  //sprintf(buff,"%s","./test.bin");
  //latfile=fopen("test.bin","r");
  //fread(randn,sizeof(double),SIZE*3,latfile);
  //fclose(latfile);

  MPI_File fh;
  MPI_Offset offset;
  int count;
 
  MPI_File_open(MPI_COMM_WORLD, "./test.bin",MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
  int n_pts = (int) SIZE/size; 
  offset = rank * n_pts * 3 * sizeof(double);
  MPI_File_read_at(fh, offset, randn, n_pts, MPI_DOUBLE, &status); 
  MPI_Get_count(&status, MPI_DOUBLE, &count); 
  printf("process %d read %d ints\n", rank, count); 
  MPI_File_close(&fh); 
  
  double min_x,min_y,min_z = DBL_MAX;
  double max_x,max_y,max_z = DBL_MIN;

  double t1 = MPI_Wtime();

  get_boundary(SIZE,randn,max_x,max_y,max_z,min_x,min_y,min_z);
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
  v->reserve(SIZE);
  pts_discrete = new std::vector<long>(SIZE * 3);
  all_inds = new std::vector<long>(SIZE);   //delete at first run of refine, see do while loop
  std::generate(all_inds->begin(), all_inds->end(), [n = 0]() mutable { return n++ ; });


  discretize(SIZE, randn, res, x_offset, y_offset, z_offset, pts_discrete);

  //std::cerr << "Discretize Elapsed time: " << elapsed1 << std::endl;
  
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
  MPI_Barrier(MPI_COMM_WORLD);
  p8est = p8est_new(mpi->mpicomm, connectivity,sizeof(user_data_t), init_fn_first, NULL);
  std::cerr << "new finished";
  p4est_gloidx_t num_quadrants = p8est->global_num_quadrants;
  //To do a breadth-first expanding, avoid using the built-in recursive approach
  //unsigned long to_destroy = 1;
  double tt = MPI_Wtime();
  do{

    num_quadrants = p8est->global_num_quadrants;
    //mpiret = sc_MPI_Barrier (mpi->mpicomm);

    p8est_refine (p8est, 0, refine_fn, init_fn);

    //MPI_Allreduce(MPI_IN_PLACE, cnt_arr, LEN, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    //mpiret = sc_MPI_Barrier (mpi->mpicomm);

    //v->erase(v->begin(), v->begin()+to_destroy);
    //to_destroy = to_destroy << 3;
    //long w = 0;
    //for (auto it = v->begin();it<v->end();it++){
    //  w += (*it)->size();
    //}
    //std::cerr << "Period size ";
    //std::cerr << w << std::endl;
  }while (num_quadrants != p8est->global_num_quadrants);
  double elapsed = MPI_Wtime() - tt;

  double elapsed1 = MPI_Wtime() - t1;
  std::cerr << "End-to-End Elapsed time: " << elapsed1 << std::endl;

  std::cerr << "Refine Elapsed time: " << elapsed << std::endl;

  mpiret = sc_MPI_Barrier (mpi->mpicomm);

  long w = 0;
  std::cerr << std::endl << v->size() << std::endl;
  for (auto it=v->begin();it<v->end();it++){
    
    //std::cerr << (*it)->size() << " ";
    if ((*it)->size() > 0){
      w+=(*it)->size();
      (*it)->clear();
      //delete (*it);
    }
    //std::cerr << (*it)->size() << " ";
  }
  //std::cerr << std::endl << w << " ";
  

  MPI_Barrier(MPI_COMM_WORLD);
  
  std::cerr << "Mem free" << std::endl;

  free(randn);
  delete pts_discrete;
  v->clear();
  delete v;

  std::cerr << "Reach 4est destroy" << std::endl;

  p8est_destroy (p8est);

  p8est_connectivity_destroy (connectivity);

  std::cerr << "Reach MPI destroy" << std::endl;
  /* clean up and exit */
  //sc_finalize ();

  //mpiret = sc_MPI_Finalize ();
  //SC_CHECK_MPI (mpiret);

  MPI_Finalize();  

  return 0;
}



void get_boundary(long data_size,double* data_ptr, double& max_x, double& max_y, double& max_z,double& min_x, double& min_y, double& min_z){
    for(long i=0;i<data_size*3;i+=3){
        if (data_ptr[i] > max_x){
            max_x = data_ptr[i];
        }

        if (data_ptr[i+1] > max_y){
            max_y = data_ptr[i+1];
        }

        if (data_ptr[i+2] > max_z){
            max_z = data_ptr[i+2];
        }

        if (data_ptr[i] < min_x){
            min_x = data_ptr[i];
        }

        if (data_ptr[i+1] < min_y){
            min_y = data_ptr[i+1];
        }

        if (data_ptr[i+2] < min_z){
            min_z = data_ptr[i+2];
        }
    }
}

void discretize(long data_size, double* data_ptr, const double &res, const int &x_offset, const int &y_offset, const int &z_offset, std::vector<long>* pts){
    double res_recp = 1.0/res;
    #pragma omp parallel for schedule(static)
    for(long j=0;j<data_size;j+=3){
        (*pts)[j] = floor(data_ptr[j] * res_recp + x_offset);   //fma
        (*pts)[j+1] = floor(data_ptr[j+1] * res_recp + y_offset);
        (*pts)[j+2] = floor(data_ptr[j+2] * res_recp + z_offset);
    }
}

