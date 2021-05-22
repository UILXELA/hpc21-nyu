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
#include <stdint.h>
#include <limits.h>


// method to seperate bits from a given integer 3 positions apart
uint64_t  splitBy3(unsigned int a){
   
  uint64_t  x = a & 0x00000000000fffff; // we only look at the first 20 bits

  x = (x | x << 32) & 0x001f00000000ffff; // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
  x = (x | x << 16) & 0x001f0000ff0000ff; // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
  x = (x | x << 8) & 0x100f00f00f00f00f; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
  x = (x | x << 4) & 0x10c30c30c30c30c3; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
  x = (x | x << 2) & 0x1249249249249249;
  return x;

}

uint64_t  mortonEncode_magicbits(unsigned int x, unsigned int y, unsigned int z){
  uint64_t  answer = 0;
  answer |= splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2;
  return answer;
}

long bin_search(const std::vector<uint64_t>& vec, const long& init_low, const long& init_high, const long& target)
{
    long low = init_low-1;
    long high = init_high+1;
    while (high-low > 1) {
        long probe = (low + high) / 2;
        if (vec[probe] >= target)
            high = probe;
        else
            low = probe;
    }

    return high;
}

#define SIZE 18833843
#define PT_COUNT 10

const int MAX_LAYERS = 19; //follows the max_level possible defined by p4est
const int MAX_REFINE_LEVEL = 17;
const long TOTAL_LENGTH = std::pow(2,MAX_LAYERS);
const long BTW_C = TOTAL_LENGTH/std::pow(2,MAX_REFINE_LEVEL+1);
const long C_LENGTH = 2*TOTAL_LENGTH-1;
const long C_LENGTHsq = std::pow(C_LENGTH,2);
//#define DBL_MAX std::numeric_limits<double>::max()
//#define DBL_MIN std::numeric_limits<double>::min()
void get_boundary(long data_size,double* data_ptr, double& max_x, double& max_y, double& max_z,double& min_x, double& min_y, double& min_z);
void discretize(long data_size, double* data_ptr, const double &res, const int &x_offset, const int &y_offset, const int &z_offset, std::vector<unsigned int>* pts);
void merge(long* X, long n, long* tmp);
void mergesort(long* X, long n, long * tmp);

long parent_lower_buf = 0;
long parent_upper_buf = 0;



template<class RanIt>
void qsort3w(RanIt _First, RanIt _Last)
{
    if (_First >= _Last) return;
    
    std::size_t _Size = 0L;
    if ((_Size = std::distance(_First, _Last)) > 0)
    {
        RanIt _LeftIt = _First, _RightIt = _Last;
        bool is_swapped_left = false, is_swapped_right = false;
        typename std::iterator_traits<RanIt>::value_type _Pivot = *_First;

        RanIt _FwdIt = _First + 1;
        while (_FwdIt <= _RightIt)
        {
            if (*_FwdIt < _Pivot)
            {
                is_swapped_left = true;
                std::iter_swap(_LeftIt, _FwdIt);
                _LeftIt++; _FwdIt++;
            }

            else if (_Pivot < *_FwdIt) {
                is_swapped_right = true;
                std::iter_swap(_RightIt, _FwdIt);
                _RightIt--;
            }

            else _FwdIt++;
        }

        if (_Size >= 1000000L)
        {
            #pragma omp taskgroup
            {
                #pragma omp task untied mergeable
                if ((std::distance(_First, _LeftIt) > 0) && (is_swapped_left))
                    qsort3w(_First, _LeftIt - 1);

                #pragma omp task untied mergeable
                if ((std::distance(_RightIt, _Last) > 0) && (is_swapped_right))
                    qsort3w(_RightIt + 1, _Last);
            }
        }

        else
        {
            #pragma omp task untied mergeable
            {
                if ((std::distance(_First, _LeftIt) > 0) && is_swapped_left)
                    qsort3w(_First, _LeftIt - 1);

                if ((std::distance(_RightIt, _Last) > 0) && is_swapped_right)
                    qsort3w(_RightIt + 1, _Last);
            }
        }
    }
}

template<class BidirIt>
void parallel_sort(BidirIt _First, BidirIt _Last)
{

    #pragma omp parallel num_threads(12)
    #pragma omp master
        qsort3w(_First, _Last - 1);
}



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
  long parent_lower;
  long parent_upper;
  long lower;
  long upper;
  long x;
  long y;
  long z;
  uint8_t level;
}user_data_t;


std::vector<user_data_t *> new_octants = std::vector<user_data_t *>();

long x_bound,y_bound,z_bound;
long lowest_count = SIZE;
//int ind;

static void my_init_fn_first (p8est_t * p4est, p4est_topidx_t which_tree,
         p8est_quadrant_t * quadrant)
{ 
  
  user_data_t        *data = (user_data_t *) quadrant->p.user_data;

  data->count = SIZE;
  data->lower = 0;
  data->upper = SIZE-1;
  data->x = quadrant->x;
  data->y = quadrant->y;
  data->z = quadrant->z;
  data->level = quadrant->level;

  new_octants.push_back(data);
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

  data->count = 0;
  data->parent_lower = parent_lower_buf;
  data->parent_upper = parent_upper_buf;
  data->x = quadrant->x;
  data->y = quadrant->y;
  data->z = quadrant->z;
  data->level = quadrant->level;


  new_octants.push_back(data);

  //std::cerr << "Child" << (int) quadrant->level << std::endl;
  //std::cerr << quadrant->x << std::endl;
  //std::cerr << quadrant->y << std::endl;
  //std::cerr << quadrant->z << std::endl;
  //std::cerr << "Level" << (int) quadrant->level << std::endl;
  //std::cerr << data->parent_upper-data->parent_lower+1 << std::endl;
  
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
    parent_lower_buf = data->lower;
    parent_upper_buf = data->upper;
    //std::cerr << "Refined:" << std::endl;
    //std::cerr << quadrant->x << std::endl;
    //std::cerr << quadrant->y << std::endl;
    //std::cerr << quadrant->z << std::endl;

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
  double* original = (double *)malloc(SIZE * 3 * sizeof(double));

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
  res =  initial_side_len / (x_bound-1);
  std::cerr << max_x << std::endl;
  std::cerr << min_x << std::endl;
  std::cerr << max_y << std::endl;
  std::cerr << min_y << std::endl;
  std::cerr << max_z << std::endl;
  std::cerr << min_z << std::endl;
  std::cerr << initial_side_len << std::endl;
  std::cerr << res << std::endl;
  std::cerr << x_bound << std::endl;
  std::cerr <<  initial_side_len/res << std::endl;

  int x_offset, y_offset, z_offset;
  x_offset = -floor(min_x/res);
  y_offset = -floor(min_y/res);
  z_offset = -floor(min_z/res);
  std::cerr << x_offset << std::endl;
  std::cerr << y_offset << std::endl;
  std::cerr << z_offset << std::endl;

  


  std::vector<unsigned int> pts_discrete = std::vector<unsigned int>(SIZE * 3);


  double t = MPI_Wtime();
  discretize(SIZE, original, res, x_offset, y_offset, z_offset, &pts_discrete);

  std::cerr << *(max_element(pts_discrete.begin(),pts_discrete.end())) << std::endl;
  std::cerr << *(min_element(pts_discrete.begin(),pts_discrete.end())) << std::endl;

  //long* bins = (long *)malloc(SIZE * sizeof(long));
  long* tmp = (long *)malloc(SIZE * sizeof(long));
  std::vector<uint64_t> bins = std::vector<uint64_t>(SIZE); 
  
  double elapsed1 = MPI_Wtime() - t;
  std::cerr << "Discretize Elapsed time: " << elapsed1 << std::endl;


  double start = omp_get_wtime();
   /*#pragma omp parallel
   {
      #pragma omp parallel for
      for(long i=0; i<SIZE; i++){
        bins[i] = C_LENGTHsq*(pts_discrete[3*i+1]/BTW_C-1)+C_LENGTH*(pts_discrete[3*i+1]/BTW_C-1)+pts_discrete[3*i+1]/BTW_C-1;
      }
      #pragma omp single`
      mergesort(bins, (long)SIZE, tmp);
   }
   */
    #pragma omp parallel
    {
      #pragma omp for
        for(long i=0; i<SIZE; i++){
          bins[i] = mortonEncode_magicbits(pts_discrete[i*3], pts_discrete[i*3+1], pts_discrete[i*3+2]);
      }
      #pragma omp master
          qsort3w(bins.begin(), bins.end()-1);
    }
    if(std::is_sorted(bins.begin(),bins.end())) std::cerr<<"Sorted" << std::endl; 
   double stop = omp_get_wtime();
   printf("\nTime: %g\n",stop-start);
   std::cerr << bins[SIZE-1] << std::endl;
   std::cerr << mortonEncode_magicbits(TOTAL_LENGTH-1, TOTAL_LENGTH-1, TOTAL_LENGTH-1) << std::endl;

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

  

  //To do a breadth-first expanding, avoid using the built-in recursive approach
  //unsigned long to_destroy = 1;




  double tt = MPI_Wtime();

  
  new_octants.reserve(SIZE);
  p4est_gloidx_t num_quadrants = p8est->global_num_quadrants;
  do{
    num_quadrants = p8est->global_num_quadrants;
    //mpiret = sc_MPI_Barrier (mpi->mpicomm);
    new_octants.clear();

    p8est_refine (p8est, 0, refine_fn, init_fn);
    
    long count = 0;
    #pragma omp parallel for schedule(static,32)
    for (long i=0; i<new_octants.size(); i++){
      uint64_t box_low;
      uint64_t box_up;
      uint64_t search_low;
      uint64_t search_high;
      user_data_t data = *(new_octants[i]);
      long side_len = x_bound >> (data.level);
      box_low = mortonEncode_magicbits((unsigned int) data.x, (unsigned int) data.y, (unsigned int) data.z);
      box_up = mortonEncode_magicbits((unsigned int) data.x+side_len-1, (unsigned int) data.y+side_len-1, (unsigned int) data.z+side_len-1);

      //std::cerr << side_len << std::endl;
      //std::cerr << new_octants[i]->x << std::endl;
      //std::cerr << new_octants[i]->y << std::endl;
      //std::cerr << new_octants[i]->z << std::endl;
      //std::cerr << std::endl;

      new_octants[i]->lower = bin_search(bins,data.parent_lower,data.parent_upper,box_low);
      new_octants[i]->upper = bin_search(bins,data.parent_lower,data.parent_upper,box_up)-1;
      new_octants[i]->count = new_octants[i]->upper-new_octants[i]->lower+1;
      //count += new_octants[i]->count; 
      //std::cerr << data->count << std::endl;
      //std::cerr << data->lower << std::endl;
      //std::cerr << data->upper << std::endl;
      //std::cerr << box_low << std::endl;
      //std::cerr << box_up << std::endl;
      //std::cerr << new_octants[i]->x << " " << new_octants[i]->y << " " << new_octants[i]->z << std::endl;
      

    }
    //std::cerr << count << std::endl;
    //std::cerr << new_octants.size() << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    
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
  
  std::cerr << "Mem free" << std::endl;

  free(original);

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

void discretize(long data_size, double* data_ptr, const double &res, const int &x_offset, const int &y_offset, const int &z_offset, std::vector<unsigned int>* pts){
    double res_recp = 1.0/res;
    #pragma omp parallel for schedule(static)
    for(long j=0;j<data_size;j+=3){
        (*pts)[j] = floor(data_ptr[j] * res_recp + x_offset);   //fma
        (*pts)[j+1] = floor(data_ptr[j+1] * res_recp + y_offset);
        (*pts)[j+2] = floor(data_ptr[j+2] * res_recp + z_offset);
    }
}

