#include "merge.h"

using namespace std;

void process_error(int severity, string err);	// this should probably live in a utils header file


#if defined(_MSC_VER)
#define BIG_CONSTANT(x) (x)
// Other compilers
#else   // defined(_MSC_VER)
#define BIG_CONSTANT(x) (x##LLU)
#endif // !defined(_MSC_VER)

unsigned int hash_seed;

struct float_avg
{
	__host__  float_type operator()(const float_type &lhs, const int_type &rhs) const {
		return lhs/rhs;
	}
};

struct float_avg1
{
	__host__  float_type operator()(const int_type &lhs, const int_type &rhs) const {
		return ((float_type)lhs)/rhs;
	}
};

struct div100
{
	__host__  int_type operator()(const int_type &lhs, const int_type &rhs) const {
		return (lhs*100)/rhs;
	}
};

thrust::host_vector<unsigned long long int> h_merge;

using namespace std;
using namespace thrust::placeholders;


void create_c(CudaSet* c, CudaSet* b)
{
	c->not_compressed = 1;
	c->segCount = 1;
	c->columnNames = b->columnNames;
	h_merge.clear();
	c->cols = b->cols;
	c->type = b->type;
	c->decimal = b->decimal;
	c->decimal_zeroes = b->decimal_zeroes;
	c->grp_type = b->grp_type;
	c->ts_cols = b->ts_cols;

	for(unsigned int i=0; i < b->columnNames.size(); i++) {
		if (b->type[b->columnNames[i]] == 0) {
			c->h_columns_int[b->columnNames[i]] = thrust::host_vector<int_type, uninitialized_host_allocator<int_type> >();
			c->d_columns_int[b->columnNames[i]] = thrust::device_vector<int_type>();
			if(b->string_map.find(b->columnNames[i]) != b->string_map.end()) {
				c->string_map[b->columnNames[i]] = b->string_map[b->columnNames[i]];
			};
		}
		else
			if (b->type[b->columnNames[i]] == 1) {
				c->h_columns_float[b->columnNames[i]] = thrust::host_vector<float_type, uninitialized_host_allocator<float_type> >();
				c->d_columns_float[b->columnNames[i]] = thrust::device_vector<float_type>();
			}
			else {
				c->h_columns_char[b->columnNames[i]] = nullptr;
				c->d_columns_char[b->columnNames[i]] = nullptr;
				c->char_size[b->columnNames[i]] = b->char_size[b->columnNames[i]];
			};
	};
}

void add(CudaSet* c, CudaSet* b, queue<string> op_v3, map<string,string> aliases,
         vector<thrust::device_vector<int_type> >& distinct_tmp, vector<thrust::device_vector<int_type> >& distinct_val,
         vector<thrust::device_vector<int_type> >& distinct_hash, CudaSet* a)
{

	if (c->columnNames.empty()) {
		// create d_columns and h_columns
		create_c(c,b);
	}

	size_t cycle_sz = op_v3.size();

	vector<string> opv;
	for(unsigned int z = 0; z < cycle_sz; z++) {
		if(std::find(b->columnNames.begin(), b->columnNames.end(), aliases[op_v3.front()]) == b->columnNames.end()) { 
			//cout << "Syntax error: alias " << op_v3.front() << endl;
			//exit(0);
			opv.push_back(op_v3.front());
		}
		else
			opv.push_back(aliases[op_v3.front()]);
		op_v3.pop();
	};


	// create hashes of groupby columns
	unsigned long long int* hashes = new unsigned long long int[b->mRecCount];
	unsigned long long int* sum = new unsigned long long int[cycle_sz*b->mRecCount];

	for(unsigned int z = 0; z < cycle_sz; z++) {
		// b->CopyColumnToHost(opv[z]);
		if(b->type[opv[z]] != 1) {  //int or string
			for(int i = 0; i < b->mRecCount; i++) {
				//memcpy(&sum[i*cycle_sz + z], &b->h_columns_int[opv[z]][i], 8);
				sum[i*cycle_sz + z] = b->h_columns_int[opv[z]][i];
				//cout << "CPY to " << i*cycle_sz + z << " " << opv[z] << " " << b->h_columns_int[opv[z]][i] <<   endl;
				//cout << "SET " << sum[i*cycle_sz + z] << endl;
			};
		}
		else {  //float
			for(int i = 0; i < b->mRecCount; i++) {
				memcpy(&sum[i*cycle_sz + z], &b->h_columns_float[opv[z]][i], 8);
			};
		};
	};

	for(int i = 0; i < b->mRecCount; i++) {
		hashes[i] = MurmurHash64A(&sum[i*cycle_sz], 8*cycle_sz, hash_seed);
		//cout << "hash " << hashes[i] << " " << i*cycle_sz << " "  << sum[i*cycle_sz] << " " << sum[i*cycle_sz + 1] << endl;
	};

	delete [] sum;
	thrust::device_vector<unsigned long long int> d_hashes(b->mRecCount);
	thrust::device_vector<unsigned int> v(b->mRecCount);
	thrust::sequence(v.begin(), v.end(), 0, 1);
	thrust::copy(hashes, hashes+b->mRecCount, d_hashes.begin());

	// sort the results by hash
	thrust::sort_by_key(d_hashes.begin(), d_hashes.end(), v.begin());

	void* d_tmp;
	CUDA_SAFE_CALL(cudaMalloc((void **) &d_tmp, b->mRecCount*int_size));

	for(unsigned int i = 0; i < b->columnNames.size(); i++) {

		if(b->type[b->columnNames[i]] == 0 || b->type[b->columnNames[i]] == 2) {
			thrust::device_ptr<int_type> d_tmp_int((int_type*)d_tmp);
			thrust::gather(v.begin(), v.end(), b->d_columns_int[b->columnNames[i]].begin(), d_tmp_int);
			thrust::copy(d_tmp_int, d_tmp_int + b->mRecCount, b->h_columns_int[b->columnNames[i]].begin());
		}
		else
			if(b->type[b->columnNames[i]] == 1) {
				thrust::device_ptr<float_type> d_tmp_float((float_type*)d_tmp);
				thrust::gather(v.begin(), v.end(), b->d_columns_float[b->columnNames[i]].begin(), d_tmp_float);
				thrust::copy(d_tmp_float, d_tmp_float + b->mRecCount, b->h_columns_float[b->columnNames[i]].begin());
			}
	};
	cudaFree(d_tmp);

	thrust::host_vector<unsigned long long int> hh = d_hashes;
	char* tmp = new char[max_char(b)*(c->mRecCount + b->mRecCount)];
	c->resize(b->mRecCount);

	//lets merge every column

	for(unsigned int i = 0; i < b->columnNames.size(); i++) {

		if(b->type[b->columnNames[i]] != 1) {

			thrust::merge_by_key(h_merge.begin(), h_merge.end(),
			                     hh.begin(), hh.end(),
			                     c->h_columns_int[c->columnNames[i]].begin(), b->h_columns_int[b->columnNames[i]].begin(),
			                     thrust::make_discard_iterator(), (int_type*)tmp);
			memcpy(thrust::raw_pointer_cast(c->h_columns_int[c->columnNames[i]].data()), (int_type*)tmp, (h_merge.size() + b->mRecCount)*int_size);
		}
		else {
			thrust::merge_by_key(h_merge.begin(), h_merge.end(),
			                     hh.begin(), hh.end(),
			                     c->h_columns_float[c->columnNames[i]].begin(), b->h_columns_float[b->columnNames[i]].begin(),
			                     thrust::make_discard_iterator(), (float_type*)tmp);
			memcpy(thrust::raw_pointer_cast(c->h_columns_float[c->columnNames[i]].data()), (float_type*)tmp, (h_merge.size() + b->mRecCount)*float_size);
		}
	};


	//merge the keys
	thrust::merge(h_merge.begin(), h_merge.end(),
	              hh.begin(), hh.end(), (unsigned long long int*)tmp);

	size_t cpy_sz = h_merge.size() + b->mRecCount;
	h_merge.resize(h_merge.size() + b->mRecCount);
	thrust::copy((unsigned long long int*)tmp, (unsigned long long int*)tmp + cpy_sz, h_merge.begin());

	delete [] tmp;
	delete [] hashes;

}

void count_avg(CudaSet* c,  vector<thrust::device_vector<int_type> >& distinct_hash)
{
	string countstr;
	thrust::equal_to<unsigned long long int> binary_pred;
	thrust::maximum<unsigned long long int> binary_op_max;
	thrust::minimum<unsigned long long int> binary_op_min;

	for(unsigned int i = 0; i < c->columnNames.size(); i++) {
		if(c->grp_type[c->columnNames[i]] == 0) { // COUNT
			countstr = c->columnNames[i];
			break;
		};
	};


	thrust::host_vector<bool> grp;
	size_t res_count;

	if(h_merge.size()) {
		grp.resize(h_merge.size());
		thrust::adjacent_difference(h_merge.begin(), h_merge.end(), grp.begin());
		res_count = h_merge.size() - thrust::count(grp.begin(), grp.end(), 0);
	};


	if (c->mRecCount != 0) {

		//unsigned int dis_count = 0;
		if (h_merge.size()) {
			int_type* tmp =  new int_type[res_count];
			for(unsigned int k = 0; k < c->columnNames.size(); k++)	{

				if(c->grp_type[c->columnNames[k]] <= 2) { //sum || avg || count
					if (c->type[c->columnNames[k]] == 0) { // int
						// check for overflow
						// convert to double, reduce, check if larger than max 64 bit int

						float_type* tmp1 =  new float_type[c->mRecCount];
						float_type* tmp_res = new float_type[res_count];

						for(int z = 0; z < c->mRecCount ; z++)
							tmp1[z] = (float_type)(c->h_columns_int[c->columnNames[k]][z]);

						thrust::reduce_by_key(h_merge.begin(), h_merge.end(), tmp1,
						                      thrust::make_discard_iterator(), tmp_res);

						double max_overflow = 0;
						for(int z = 0; z < res_count; z++) {
							if (tmp_res[z] > 9223372036854775807.0) {
								if(tmp_res[z] - 9223372036854775807.0 > max_overflow)
									max_overflow = tmp_res[z];
							};
						};
						if(max_overflow) {
							unsigned pw = ceil(log10(max_overflow/9223372036854775807.0));
							thrust::transform(c->h_columns_int[c->columnNames[k]].begin(), c->h_columns_int[c->columnNames[k]].end(), thrust::make_constant_iterator((int_type)pow(10, pw)), c->h_columns_int[c->columnNames[k]].begin(), thrust::divides<int_type>());
							c->decimal_zeroes[c->columnNames[k]] = c->decimal_zeroes[c->columnNames[k]] - pw;
						};

						delete [] tmp1;
						delete [] tmp_res;

						thrust::reduce_by_key(h_merge.begin(), h_merge.end(), c->h_columns_int[c->columnNames[k]].begin(),
						                      thrust::make_discard_iterator(), tmp);
						c->h_columns_int[c->columnNames[k]].resize(res_count);
						thrust::copy(tmp, tmp + res_count, c->h_columns_int[c->columnNames[k]].begin());
					}
					else
						if (c->type[c->columnNames[k]] == 1 ) { // float
							float_type* tmp1 =  new float_type[res_count];
							thrust::reduce_by_key(h_merge.begin(), h_merge.end(), c->h_columns_float[c->columnNames[k]].begin(),
							                      thrust::make_discard_iterator(), tmp1);
							c->h_columns_float[c->columnNames[k]].resize(res_count);
							thrust::copy(tmp1, tmp1 + res_count, c->h_columns_float[c->columnNames[k]].begin());
							delete [] tmp1;
						};
				}
				if(c->grp_type[c->columnNames[k]] == 4) { //min
					if (c->type[c->columnNames[k]] == 0 ) { // int
						thrust::reduce_by_key(h_merge.begin(), h_merge.end(), c->h_columns_int[c->columnNames[k]].begin(),
						                      thrust::make_discard_iterator(), tmp, binary_pred, binary_op_min);
						c->h_columns_int[c->columnNames[k]].resize(res_count);
						thrust::copy(tmp, tmp + res_count, c->h_columns_int[c->columnNames[k]].begin());
					}
					else
						if (c->type[c->columnNames[k]] == 1 ) { // float
							c->h_columns_float[c->columnNames[k]].resize(res_count);
							thrust::reduce_by_key(h_merge.begin(), h_merge.end(), c->h_columns_float[c->columnNames[k]].begin(),
							                      thrust::make_discard_iterator(), c->h_columns_float[c->columnNames[k]].begin(), binary_pred, binary_op_min);
						};
				}
				if(c->grp_type[c->columnNames[k]] == 5) { //max
					if (c->type[c->columnNames[k]] == 0 ) { // int
						int_type* tmp =  new int_type[res_count];
						thrust::reduce_by_key(h_merge.begin(), h_merge.end(), c->h_columns_int[c->columnNames[k]].begin(),
						                      thrust::make_discard_iterator(), tmp, binary_pred, binary_op_max);
						c->h_columns_int[c->columnNames[k]].resize(res_count);
						thrust::copy(tmp, tmp + res_count, c->h_columns_int[c->columnNames[k]].begin());
						delete [] tmp;
					}
					else
						if (c->type[c->columnNames[k]] == 1 ) { // float
							c->h_columns_float[c->columnNames[k]].resize(res_count);
							thrust::reduce_by_key(h_merge.begin(), h_merge.end(), c->h_columns_float[c->columnNames[k]].begin(),
							                      thrust::make_discard_iterator(), c->h_columns_float[c->columnNames[k]].begin(), binary_pred, binary_op_max);
						};
				}
				else
					if(c->grp_type[c->columnNames[k]] == 3) { //no group function
						if (c->type[c->columnNames[k]] == 0 || c->type[c->columnNames[k]] == 2) { // int
							thrust::reduce_by_key(h_merge.begin(), h_merge.end(), c->h_columns_int[c->columnNames[k]].begin(),
							                      thrust::make_discard_iterator(), tmp, binary_pred, binary_op_max);
							c->h_columns_int[c->columnNames[k]].resize(res_count);
							thrust::copy(tmp, tmp + res_count, c->h_columns_int[c->columnNames[k]].begin());
						}
						else
							if (c->type[c->columnNames[k]] == 1 ) { // float
								c->h_columns_float[c->columnNames[k]].resize(res_count);
								thrust::reduce_by_key(h_merge.begin(), h_merge.end(), c->h_columns_float[c->columnNames[k]].begin(),
								                      thrust::make_discard_iterator(), c->h_columns_float[c->columnNames[k]].begin(), binary_pred, binary_op_max);
							}
					};
			};
			c->mRecCount = res_count;
			delete [] tmp;
		};

		for(unsigned int k = 0; k < c->columnNames.size(); k++)	{
			if(c->grp_type[c->columnNames[k]] == 1) {   // AVG

				if (c->type[c->columnNames[k]] == 0 ) { // int

					if(c->decimal_zeroes[c->columnNames[k]] <= 2) {
						thrust::transform(c->h_columns_int[c->columnNames[k]].begin(), c->h_columns_int[c->columnNames[k]].begin() + c->mRecCount,
						                  c->h_columns_int[countstr].begin(), c->h_columns_int[c->columnNames[k]].begin(), div100());
						c->decimal_zeroes[c->columnNames[k]] = c->decimal_zeroes[c->columnNames[k]] + 2;
					}
					else {
						thrust::transform(c->h_columns_int[c->columnNames[k]].begin(), c->h_columns_int[c->columnNames[k]].begin() + c->mRecCount,
						                  c->h_columns_int[countstr].begin(), c->h_columns_int[c->columnNames[k]].begin(), thrust::divides<int_type>());
					};
					c->grp_type[c->columnNames[k]] = 3;
				}
				else {              // float
					thrust::transform(c->h_columns_float[c->columnNames[k]].begin(), c->h_columns_float[c->columnNames[k]].begin() + c->mRecCount,
					                  c->h_columns_int[countstr].begin(), c->h_columns_float[c->columnNames[k]].begin(), float_avg());
				};
			}
			else
				if(c->grp_type[c->columnNames[k]] == 6) {
				}
				else
					if(c->grp_type[c->columnNames[k]] == 2) {

					};
		};

	};

	c->segCount = 1;
	c->maxRecs = c->mRecCount;
};

