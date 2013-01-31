
#define MOVEBITS 22 			//Using 22 bits for Supervertex IDs, hope this is enough, leaving 10 bits for weights, max 1K weight range
#define APPEND_VERTEX_BITS 32
#define INF 100000000
#define MAX_THREADS_PER_BLOCK 512


__global__ void makeVertexList(unsigned int *d_vertices, unsigned int *d_edge_flag, int *d_pick_array, int num_edges)
{
	unsigned int i = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
//	for(int i=0; i<num_edges; i++)
	if(i < num_edges)
	{
		if(d_edge_flag[i]==1)
		{
			unsigned int pos=d_pick_array[i]; //get the u value
			d_vertices[pos]=i; //write the index to the u'th value in the array to create the vertex list
		}	
	}
}


__global__ void makeFlagArrayForNewVertexList(unsigned int *d_edge_flag, int *d_pick_array, int num_edges)
{
	unsigned int i = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
//	for(int i=0; i<num_edges;i++)
	if( i < num_edges)
	{
		if(i==0)
		{
			d_edge_flag[i]=1;
		}
		else
		{
			if(d_pick_array[i-1]<d_pick_array[i])
				d_edge_flag[i]=1;
		}
	}
}

__global__ void copyEdgeMapping(unsigned int *d_edge_mapping_copy, unsigned int *d_edge_mapping, 
						unsigned int *d_weights, unsigned int *d_edges,
						unsigned int *d_segmented_min_scan_input, unsigned int *d_segmented_min_scan_output,  
						unsigned int *new_edges)
{
	unsigned int i = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
//	for(int i=0; i<*new_edges;i++)
	if(i < *new_edges)
	{
		d_edge_mapping[i] = d_edge_mapping_copy[i]; 
		d_edges[i] = d_segmented_min_scan_input[i]; 
		d_weights[i] = d_segmented_min_scan_output[i]; 
	}
}

__global__ void compactEdgeListDuplicates(unsigned int *d_edges, unsigned int *d_weights, unsigned int *d_new_vertex_Ids, 
								unsigned int *d_edge_mapping, unsigned int *d_edge_mapping_copy, 
								unsigned int *d_segmented_min_scan_input, unsigned int *d_segmented_min_scan_output, 
								unsigned int *d_edge_flag, unsigned long long int *d_appended_uindex, 
								int *d_pick_array, unsigned int *new_edges, unsigned int *new_vertex_size,
								unsigned int *new_edge_size)
{
	unsigned int i = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	
//	for(int i=0; i<*new_edges; i++)
	if(i < *new_edges)
	{
		unsigned long long int val = d_appended_uindex[i]; 
		unsigned long long int mask = pow(2.0, APPEND_VERTEX_BITS)-1;
		unsigned long long int index = val&mask;
		unsigned long long int u = val >> APPEND_VERTEX_BITS;
		unsigned int v = d_edges[index];
		if(u!=INF && v!=INF)
		{
			//Copy the edge_mapping into a temporary array, used to resolve read after write inconsistancies
			d_edge_mapping_copy[i] = d_edge_mapping[index]; //keep a mapping from old edge-list to new one
			d_pick_array[i]=u; // reusing this to store u's
			d_segmented_min_scan_output[i]= d_weights[index]; //resuing d_segmented_min_scan_output to store weights
			d_segmented_min_scan_input[i] = d_new_vertex_Ids[v]; //resuing d_segmented_scan_input to store v ids
			//Set the new vertex list and edge list sizes
			if(i==*new_edges-1)
			{
				*new_edge_size=(i+1);
				*new_vertex_size=(u+1);
			}
		}
	}
}


__global__ void markEdgesNew(unsigned int *d_edge_flag, unsigned long long int *d_appended_uindex, 
					unsigned int *d_new_edges, int num_edges)
{
	unsigned int i = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
//	for(int i=0; i< num_edges; i++)
	if(i < num_edges)
	{
		if(i==0)
			d_edge_flag[i]=1;
		else
		{
			unsigned long long int prev = d_appended_uindex[i-1]>>APPEND_VERTEX_BITS;
			unsigned long long int curr = d_appended_uindex[i]>>APPEND_VERTEX_BITS;
			if(curr > prev)
				d_edge_flag[i]=1;
			if(curr == INF && prev != INF)
			{
				*d_new_edges = i;
			}
		}
	}
}


__global__ void sortArray(unsigned long long int *d_array, int length)
{
	unsigned int i = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	unsigned long long int Temp;
	int j;
//	for(int i=1; i<length; i++)
	if (i < length)
	{
		Temp = d_array[i];
		j = i-1;
		while(Temp<d_array[j] && j>=0)
		{
			d_array[j+1] = d_array[j];
			j = j-1;
		}
		d_array[j+1] = Temp;
	}
}


__global__ void appendForNoDuplicateEdgeRemoval(unsigned long long int *d_appended_uindex, unsigned int *d_edges, unsigned int *d_old_uId,
										 unsigned int *d_weights, unsigned int *d_new_vertex_Ids, int num_edges)
{
	unsigned int i = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
//	for(int i=0; i< num_edges; i++)
	if(i < num_edges)
	{
		unsigned long long int val;
		unsigned int u,v,superuid=INF;
		u = d_old_uId[i];
		v = d_edges[i];
		if(u!= INF && v!=INF)
		{
			superuid = d_new_vertex_Ids[u];	
		}
		val = superuid;
		val = val<<APPEND_VERTEX_BITS;
		val |= i;
		d_appended_uindex[i] = val;
	}
}


__global__ void removeSelfEdges(unsigned int *d_edges, unsigned int *d_old_uId, 
						unsigned int *d_new_vertex_Ids, int num_edges)
{
	unsigned int i = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
//	for(int i=0; i< num_edges; i++)
	if(i < num_edges)
	{
		unsigned int u = d_old_uId[i];
		unsigned int v = d_edges[i];
		unsigned int u_parent = d_new_vertex_Ids[u];
		unsigned int v_parent = d_new_vertex_Ids[v];
		if(u_parent == v_parent)
		{
			d_edges[i] = INF;
		}
	}
}

__global__ void markSuperVertexIdsPerVertex(unsigned int *d_new_vertex_Ids, unsigned long long int *d_append_Ids,
									unsigned int *d_vertex_flag, int num_vertices)
{
	unsigned int i = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
//	for(int i=0; i< num_vertices; i++)
	if(i < num_vertices)
	{
		unsigned long long int mask = pow(2.0, APPEND_VERTEX_BITS)-1;
		unsigned long long int vertexid = d_append_Ids[i]&mask;
		d_vertex_flag[vertexid] = d_new_vertex_Ids[i];
	}
}


__global__ void markVertexFlag(unsigned int *d_vertex_flag, unsigned long long int *d_append_Ids, int num_vertices)
{
	unsigned int i = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
//	for(int i=0; i< num_vertices; i++)
	if(i < num_vertices)
	{
		if(i > 0)
		{
			unsigned int parent_curr = d_append_Ids[i] >> APPEND_VERTEX_BITS;
			unsigned int parent_prev = d_append_Ids[i-1] >> APPEND_VERTEX_BITS;
			if( parent_curr != parent_prev)
				d_vertex_flag[i]=1;
		}
		 
	}
}


__global__ void sortSuperVertexIds(unsigned long long int *d_append_Ids, int num_vertices)
{
	unsigned int i = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	unsigned long long int Temp;
	int j;
//	for(int i=1; i<num_vertices; i++)
	if(i < num_vertices)
	{
		Temp = d_append_Ids[i];
		j = i-1;
		while(Temp<d_append_Ids[j] && j>=0)
		{
			d_append_Ids[j+1] = d_append_Ids[j];
			j = j-1;
		}
		d_append_Ids[j+1] = Temp;
	}
}


__global__ void appendVertexId(unsigned long long int *d_append_Ids, unsigned int *d_successor, int num_vertices)
{
	unsigned int i = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
//	for (int i=0; i< num_vertices; i++)
	if(i < num_vertices)
	{
		unsigned long long int parent = d_successor[i];
		parent = parent << APPEND_VERTEX_BITS;
		parent |= i;
		d_append_Ids[i] = parent;
	}
}

__global__ void doPointerDoubling(unsigned int *d_successor,unsigned int *d_successor_copy, int num_vertices, bool *d_change)
{
	unsigned int i = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
//	for(int i=0; i< num_vertices; i++)
	if(i < num_vertices)
	{
		unsigned int parent = d_successor[i];
		unsigned int grandParent = d_successor[parent];
		if( grandParent != parent)
		{
			d_successor_copy[i] = grandParent;
			*d_change = true;
		}
	}
}

__global__ void copyArray(unsigned int *newArray, unsigned int *oldArray, int num_elements)
{
	unsigned int i = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
//	for(int i =0; i< num_elements; i++)
	if(i < num_elements)
	{
		newArray[i] = oldArray[i];
	}
}


__global__ void markOutputEdges(int *d_pick_array, unsigned int *d_segmented_min_scan_output, 
						unsigned int *d_output_MST, unsigned int *d_edge_mapping, int num_edges)
{
	unsigned int i = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
//	for(int i=0; i<num_edges; i++)
	if(i < num_edges)
	{
		int index = d_pick_array[i];
		if( index >= 0)
		{
			unsigned int output = d_segmented_min_scan_output[index];
			unsigned int curr = d_segmented_min_scan_output[i];
			unsigned int prev = d_segmented_min_scan_output[i-1];
			int prev_index = d_pick_array[i-1];
			if(prev_index == index) 
			{
				if(curr == output && curr != prev) 
				{
					unsigned int edgeid = d_edge_mapping[i];
					d_output_MST[edgeid]=1;
				}
			}
			else
			{
				if(curr == output)
				{
					unsigned int edgeid = d_edge_mapping[i];
					d_output_MST[edgeid]=1;
				}

			}
		}
	}
}


__global__ void initialiseArray(unsigned int *d_edge_mapping, int num_edges)
{
	unsigned int i = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
//	for(int i =0; i< num_edges; i++)
	if(i < num_edges)
	{
		d_edge_mapping[i] = i;
	}
}

__global__ void makePickArray(int *d_pick_array, unsigned int *d_successor,unsigned int *d_vertices,
					unsigned int *d_old_uID,int num_vertices, int num_edges)
{
	unsigned int i = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
//	for(int i =0; i< num_edges; i++)
	if(i < num_edges)
	{
		unsigned int parent = d_old_uID[i];
		unsigned int end=0;
		if( parent < ((num_vertices) - 1))
		{
			end = d_vertices[parent+1] -1;
		}
		else
		{
			end = num_edges-1;
		}
		if( parent != d_successor[parent])
			d_pick_array[i] = end;
		else
			{
				d_pick_array[i] = -1;
			}
	}
}

__global__ void modifyOldUID(unsigned int *d_old_uID, int num_edges)
{
	unsigned int i = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(i < num_edges)
	{
		d_old_uID[i] = d_old_uID[i]-1;
	}
}
__global__ void prefixSum(unsigned int *d_old_uID, unsigned int *d_edge_flag, int num_edges)
{
	unsigned int i = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
//	for (int i=0;i<num_edges; i++)
	if(i < num_edges)
	{
		if(i == 0)
		//	d_old_uID[i]= d_edge_flag[i];
			d_old_uID[i]= 0;
		else
			d_old_uID[i]= d_old_uID[i-1]+d_edge_flag[i];
	}
}

__global__ void markFlagForUid(unsigned int *d_edge_flag, unsigned int *d_vertices, int num_vertices)
{
	unsigned int i = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
//	for(int i=0; i< num_vertices; i++)
	if(i < num_vertices)
	{
		unsigned int vertex = d_vertices[i];
		d_edge_flag[vertex]=1;	
	}
}

__global__ void removeCycles(unsigned int *d_successor, int num_vertices)
{
	unsigned int i = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
//	for(int i =0; i< num_vertices; i++)
	if(i < num_vertices)
	{
		unsigned int succ = d_successor[i];
		unsigned int nextsucc = d_successor[succ];
		if(i == nextsucc)
		{
			if(i < succ)
				d_successor[i]=i;
			else
				d_successor[succ]=succ;
		}
	}
}



__global__ void makeSuccesorArray(unsigned int *d_successor, unsigned int *d_vertices, unsigned int *d_segmented_min_scan_output, 
					    int num_vertices, int num_edges)
{
	unsigned int i = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
//	for (int i =0; i< num_vertices; i++)
	if(i < num_vertices)
	{
		unsigned int end;
		if(i < (num_vertices-1))
			end = d_vertices[i+1]-1;
		else
			end = num_edges-1;
		unsigned int mask = pow(2.0, MOVEBITS)-1;
		d_successor[i] = d_segmented_min_scan_output[end]&mask;
	}
}

__device__ unsigned int min_device(unsigned int a, unsigned int b)
{
	int tmp = a;
	if(a > b)
	{
		tmp = b;
	}
	return tmp;
}

__global__ void segmentedMinScan(unsigned int *d_segmented_min_scan_output, unsigned int *d_segmented_min_scan_input,
					  const unsigned int *d_edge_flag, int num_edges)
{
	unsigned int i = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
//	for(int i=0; i< num_edges; i++)
	if(i < num_edges)
	{
		if(d_edge_flag[i] == 1)
		{
			d_segmented_min_scan_output[i]= d_segmented_min_scan_input[i];
		}
		else
		{
			d_segmented_min_scan_output[i] = min_device(d_segmented_min_scan_output[i-1],d_segmented_min_scan_input[i]);
		}
	}
	
}



__global__ void appendWeight(unsigned int *d_segmented_min_scan_input, unsigned int *d_weights, unsigned int *d_edges, int num_edges)
{
	unsigned int i = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
//	for (int i=0; i< num_edges; i++)
	if(i < num_edges)
	{
		unsigned int val=d_weights[i];
		val=val<<MOVEBITS;
		val|=d_edges[i];
		d_segmented_min_scan_input[i]=val;
	}
}

/*
__global__ void printArray(unsigned int *arrayToBePrint, int num_elements)
{
	unsigned int i = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
//	for(int i=0; i< num_elements; i++)
	if(i < num_elements)
	{
		cout<<arrayToBePrint[i]<<" ";	
	}
	cout<<"\n";
}

__global__ void printArrayInt(int *arrayToBePrint, int num_elements)
{
	unsigned int i = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
//	for(int i=0; i< num_elements; i++)
	if(i < num_elements)
	{
		cout<<arrayToBePrint[i]<<" ";	
	}
	cout<<"\n";
}
*/

__global__ void clearArray(unsigned int *array, int num_elements)
{
	unsigned int i = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
//	for(int i=0; i< num_elements; i++)
	if(i < num_elements)
	{
		array[i] = 0;	
	}
}

__global__ void clearArrayInt(int *array, int num_elements)
{
	unsigned int i = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
//	for(int i=0; i< num_elements; i++)
	if( i < num_elements)
	{
		array[i] = 0;	
	}
}

__global__ void markEdgeFlag(unsigned int *d_edge_flag, unsigned int *d_vertices, int num_vertices)
{
	unsigned int i = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
//	for(int i=0; i< num_vertices; i++)
	if(i < num_vertices)
	{
		unsigned int vertex = d_vertices[i];
		d_edge_flag[vertex]=1;	
	}
}
