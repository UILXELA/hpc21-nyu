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
#include <tuple>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <unordered_map>

//#define SIZE 18833843
#define SIZE 18833843
#define PT_COUNT 10

const int MAX_LAYERS = 19; //follows the max_level possible defined by p4est
const int MAX_REFINE_LEVEL = 19;
const long TOTAL_LENGTH = std::pow(2,MAX_LAYERS);
const long BTW_C = TOTAL_LENGTH/std::pow(2,MAX_REFINE_LEVEL+1);
const long C_LENGTH = 2*TOTAL_LENGTH-1;
const long C_LENGTHsq = std::pow(C_LENGTH,2);
//#define DBL_MAX std::numeric_limits<double>::max()
//#define DBL_MIN std::numeric_limits<double>::min()
void get_boundary(long data_size,double* data_ptr, double& max_x, double& max_y, double& max_z,double& min_x, double& min_y, double& min_z);
void discretize(long data_size, double* data_ptr, const double &res, const int &x_offset, const int &y_offset, const int &z_offset, std::vector<long>* pts);
void reduce_umaps(\
    std::unordered_map<long, long>& output, \
    std::unordered_map<long, long>& input)
{
    for (auto& X : input) {
      output.at(X.first) += X.second; //Will throw if X.first doesn't exist in output. 
    }
}
#pragma omp declare reduction(umap_reduction : \
    std::unordered_map<long, long> : \
    reduce_umaps(omp_out, omp_in)) \
    initializer(omp_priv(omp_orig))

typedef struct
{
  sc_MPI_Comm         mpicomm;
  int                 mpisize;
  int                 mpirank;
}
mpi_context_t;

typedef struct
{
  //std::vector<long>* ind_vec_ptr;
  long count;
  long* fused_count;
  int c_x;
  int c_y;
  int c_z;
  long bin_ind;
  uint8_t level;
}user_data_t;



std::vector<std::vector<long>*>* v;
std::vector<long>* v_count;

std::unordered_map<long,user_data_t*> octants_map;  //key:center

long x_bound,y_bound,z_bound;
long lowest_count = SIZE;
//int ind;

static void my_init_fn_first (p8est_t * p4est, p4est_topidx_t which_tree,
         p8est_quadrant_t * quadrant)
{ 
  user_data_t        *data = (user_data_t *) quadrant->p.user_data;

  data->count = SIZE;
  data->c_x = (int)x_bound/2;
  data->c_y = (int)y_bound/2;
  data->c_z = (int)z_bound/2;
  data->bin_ind = C_LENGTHsq*(data->c_x/BTW_C-1)+C_LENGTH*(data->c_y/BTW_C-1)+(data->c_z/BTW_C-1);
  //octants_map[]
  data->level = quadrant->level;

  octants_map[data->bin_ind] = data;

  //std::cerr << quadrant->x << std::endl;
  //std::cerr << quadrant->y << std::endl;
  //std::cerr << quadrant->z << std::endl;
  //std::cerr << "Level" << (int) quadrant->level << std::endl;
  //std::cerr << data2->c_x << " ";

  return;
  //std::vector<long>* data = (std::vector<long>*) quadrant->p.user_data;
}

static void my_init_fn (p8est_t * p4est, p4est_topidx_t which_tree,
         p8est_quadrant_t * quadrant)
{
  user_data_t        *data = (user_data_t *) quadrant->p.user_data;

  int half_side_len = x_bound >> (quadrant->level+1);
  //std::cerr<<half_side_len << std::endl;
  
  data->c_x = quadrant->x+half_side_len;
  data->c_y = quadrant->y+half_side_len;
  data->c_z = quadrant->z+half_side_len;
  data->count = 0;
  data->bin_ind = C_LENGTHsq*(data->c_x/BTW_C-1)+C_LENGTH*(data->c_y/BTW_C-1)+(data->c_z/BTW_C-1);
  //octants_map[]
  data->level = quadrant->level;

  octants_map[data->bin_ind] = data;

  
  /*
  std::cerr<< "Bin ind "<< data->bin_ind<< std::endl;
  std::cerr << data->c_x << " ";
  std::cerr << data->c_y << " ";
  std::cerr << data->c_z << " ";
  std::cerr << C_LENGTHsq << " ";
  std::cerr << C_LENGTH << " ";
  std::cerr << BTW_C << " ";
  std::cerr << (data->c_x/BTW_C-1) << " ";
  std::cerr << std::endl;
  */
  
  return;
  //std::vector<long>* data = (std::vector<long>*) quadrant->p.user_data;
}


static int
my_refine_fn (p8est_t * p4est, p4est_topidx_t which_tree,
                p8est_quadrant_t * quadrant)
{

  
  if (quadrant->level >= MAX_REFINE_LEVEL){
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
    octants_map.erase(data->bin_ind);    

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

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Rank %d/%d running on %s.\n", rank, size, processor_name);

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
  double* original = (double *)malloc(SIZE * 3 * sizeof(double));

  //char buff[256];
  /*
  FILE *latfile;

  //sprintf(buff,"%s","./test.bin");
  latfile=fopen("test.bin","r");
  fread(original,sizeof(double),SIZE*3,latfile);
  fclose(latfile);
  */
  MPI_File fh;
  MPI_Offset offset;
  int count;
 
  MPI_File_open(MPI_COMM_WORLD, "./test.bin",MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
  int n_pts = (int) SIZE/size; 
  offset = rank * n_pts * 3 * sizeof(double);
  MPI_File_read_at(fh, offset, original, n_pts, MPI_DOUBLE, &status); 
  MPI_Get_count(&status, MPI_DOUBLE, &count); 
  printf("process %d read %d ints\n", rank, count); 
  MPI_File_close(&fh); 

  
  double min_x,min_y,min_z = DBL_MAX;
  double max_x,max_y,max_z = DBL_MIN;

  get_boundary(SIZE,original,max_x,max_y,max_z,min_x,min_y,min_z);
  double initial_side_len;
  double res;
  initial_side_len = std::max({max_x-min_x, max_y-min_y, max_z-min_z});
  x_bound = y_bound = z_bound = TOTAL_LENGTH;
  res =  initial_side_len / x_bound;
  /*
  std::cerr << max_x << std::endl;
  std::cerr << min_x << std::endl;
  std::cerr << max_y << std::endl;
  std::cerr << min_y << std::endl;
  std::cerr << max_z << std::endl;
  std::cerr << min_z << std::endl;
  std::cerr << initial_side_len << std::endl;
  std::cerr << res << std::endl;
  std::cerr << x_bound << std::endl;
  */
  int x_offset, y_offset, z_offset;
  x_offset = -floor(min_x/res);
  y_offset = -floor(min_y/res);
  z_offset = -floor(min_z/res);
  //std::cerr << x_offset << std::endl;
  //std::cerr << y_offset << std::endl;
  //std::cerr << z_offset << std::endl;

  
  std::vector<long> pts_discrete = std::vector<long>(SIZE * 3);

  octants_map.reserve(SIZE);

  double t = MPI_Wtime();
  discretize(SIZE, original, res, x_offset, y_offset, z_offset, &pts_discrete);
  double elapsed1 = MPI_Wtime() - t;
  //std::cerr << "Discretize Elapsed time: " << elapsed1 << std::endl;
  
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
  //std::cerr << "new finished";


  std::vector<int> pt2bin_map_x = std::vector<int>(SIZE,(*(octants_map.begin())).second->c_x);   //follow the order of pts, save the bin_id in coord and bin to avoid recalc
  std::vector<int> pt2bin_map_y = std::vector<int>(SIZE,(*(octants_map.begin())).second->c_y);
  std::vector<int> pt2bin_map_z = std::vector<int>(SIZE,(*(octants_map.begin())).second->c_z);
  std::vector<uint8_t> pt2bin_map_level = std::vector<uint8_t>(SIZE,1);
  

  //To do a breadth-first expanding, avoid using the built-in recursive approach
  //unsigned long to_destroy = 1;




  double tt = MPI_Wtime();

  std::unordered_map<long,long> cnts_map;
  cnts_map.reserve(SIZE);
  p4est_gloidx_t num_quadrants = p8est->global_num_quadrants;
  do{
    num_quadrants = p8est->global_num_quadrants;
    //mpiret = sc_MPI_Barrier (mpi->mpicomm);
    cnts_map.clear();

    p8est_refine (p8est, 0, refine_fn, init_fn);

    
    
    //cnts_map.reserve(octants_map.size());
    //#pragma omp parallel for
    for(auto it = octants_map.begin(); it != octants_map.end(); ++it){
      cnts_map[((it)->first)] = 0;
      //std::cerr << ((it)->first) << std::endl;
    }
    //std::cerr << "map size" << octants_map.size() << std::endl;
    //std::cerr << "cnt size" << cnts_map.size() << std::endl;

    //std::cerr<<"count init Done" << std::endl;

    #pragma omp parallel
    {
      
    #pragma omp for schedule(static,4096) //reduction(umap_reduction:cnts_map)
    for(long i=0;i<SIZE;i++){
      long bin_id = C_LENGTHsq*(pt2bin_map_x[i]/BTW_C-1)+C_LENGTH*(pt2bin_map_y[i]/BTW_C-1)+pt2bin_map_z[i]/BTW_C-1;
      int half_side_len = x_bound >> (pt2bin_map_level[i]+1);
      if (octants_map.find(bin_id) == octants_map.end()){
        //std::cerr<<"Entered loop" << std::endl;
        //assign new bin
        long x,y,z;
        long i3 = i*3;
        if (pts_discrete[i3] <= pt2bin_map_x[i]){
          pt2bin_map_x[i]-=half_side_len;
        }else{
          pt2bin_map_x[i]+=half_side_len;
        }  
        if (pts_discrete[i3+1] <= pt2bin_map_y[i]){
          pt2bin_map_y[i]-=half_side_len;
        }else{
          pt2bin_map_y[i]+=half_side_len;
        } 
        if (pts_discrete[i3+2] <= pt2bin_map_z[i]){
          pt2bin_map_z[i]-=half_side_len;
        }else{
          pt2bin_map_z[i]+=half_side_len;
        }

        //update count
        //bin_id = C_LENGTHsq*(pt2bin_map_x[i]/BTW_C-1)+C_LENGTH*(pt2bin_map_y[i]/BTW_C-1)+pt2bin_map_z[i]/BTW_C-1;
        //cnts_map[bin_id]+=1;
        //std::cerr << cnts_map[bin_id] << std::endl;
        pt2bin_map_level[i]+=1;
      }else{
        //cnts_map[bin_id] = octants_map[bin_id]->count;
      }
    }
    //std::cerr << "Exit main loop " << std::endl;
    
    #pragma omp for schedule(static,1024) reduction(umap_reduction:cnts_map)
    for(long i=0;i<SIZE;i++){
      long bin_id = C_LENGTHsq*(pt2bin_map_x[i]/BTW_C-1)+C_LENGTH*(pt2bin_map_y[i]/BTW_C-1)+pt2bin_map_z[i]/BTW_C-1;

      cnts_map[bin_id]+=1;

    }
    
    //std::cerr << "Exit count loop " << std::endl;

    //std::cerr<<"Exited loop" << std::endl;
    //long w=0;
    #pragma omp for
    for(size_t b=0;b<cnts_map.bucket_count();b++){
    for(auto it = cnts_map.begin(b); it != cnts_map.end(b); ++it){
      //std::cerr << (it)->first << std::endl;
      octants_map[((it)->first)]->count = ((it)->second);
      //w+=((it)->second);
      
    }  
    }
    }

    //std::cerr << "Period size ";
    //std::cerr << w << std::endl;

    
    /*
    int i=0;
    
    for(auto const & it: octants_map){
      i++;
      std::cerr << ((it).second)->c_x << "  ";
      std::cerr << ((it).second)->c_y << "  ";
      std::cerr << ((it).second)->c_z << " " << i << std::endl;
    }
    */



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

  std::cerr << "Refine Elapsed time: " << elapsed << std::endl;

  //mpiret = sc_MPI_Barrier (mpi->mpicomm);


  //std::cerr << std::endl << w << " ";
  

  MPI_Barrier(MPI_COMM_WORLD);
  
  //std::cerr << "Mem free" << std::endl;

  free(original);

  //std::cerr << "Reach 4est destroy" << std::endl;

  p8est_destroy (p8est);

  p8est_connectivity_destroy (connectivity);

  //std::cerr << "Reach MPI destroy" << std::endl;
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

