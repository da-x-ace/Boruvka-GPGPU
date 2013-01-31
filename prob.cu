#include<iostream>
#include <cuda.h>
#include <math.h>
#include <sys/types.h>
#include <time.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cutil.h>
#include <cudpp.h>
#include <sys/types.h>
#include <time.h>
#include "cudpp_config.h"
#include "functions.cu"
#define MOVEBITS 22 			//Using 22 bits for Supervertex IDs, hope this is enough, leaving 10 bits for weights, max 1K weight range
#define APPEND_VERTEX_BITS 32
#define INF 100000000
#define MAX_THREADS_PER_BLOCK 512

using namespace std;

//readFile
unsigned int num_vertices, num_edges;
unsigned int num_vertices_orig, num_edges_orig;
unsigned int *h_vertices, *h_length, *h_edges, *h_weights;
unsigned int *d_vertices, *d_length, *d_edges, *d_weights;
unsigned int *edges_orig, *weights_orig;
int source;
int MSTWeight=0;
int iteration =0;

//MST
unsigned int *h_segmented_min_scan_input,*h_segmented_min_scan_output;
unsigned int *d_segmented_min_scan_input,*d_segmented_min_scan_output;

unsigned int *h_edge_flag, *h_vertex_flag, *h_old_uID;
unsigned int *d_edge_flag, *d_vertex_flag, *d_old_uID;

unsigned int *h_successor, *h_successor_copy;
unsigned int *d_successor, *d_successor_copy;

int *h_pick_array;
int *d_pick_array;

unsigned int *h_output_MST;
unsigned int *d_output_MST;

unsigned int *h_edge_mapping, *h_edge_mapping_copy;
unsigned int *d_edge_mapping, *d_edge_mapping_copy;

unsigned long long int *h_append_Ids, *h_appended_uindex;
unsigned long long int *d_append_Ids, *d_appended_uindex;

unsigned int *h_new_vertex_Ids;
unsigned int *d_new_vertex_Ids;

bool change;
bool *d_change;

unsigned int new_edges;
unsigned int *d_new_edges;

unsigned int new_vertex_size, new_edge_size;
unsigned int *d_new_vertex_size, *d_new_edge_size;

//CUDPP Scan and Segmented Scan Variables
CUDPPHandle		theCudpp,segmentedScanPlan_min=0, scanPlan_add =0; 
CUDPPConfiguration	config_segmented_min, config_scan_add ;

// Functions 

void sortArrayHost(unsigned long long int *d_array, int length)
{
	unsigned long long int Temp;
	int j;
	for(int i=1; i<length; i++)
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

void printArrayHost(unsigned int *arrayToBePrint, int num_elements)
{
	for(int i=0; i< num_elements; i++)
	{
		cout<<arrayToBePrint[i]<<" ";	
	}
	cout<<"\n";
}

void printArrayHostInt(int *arrayToBePrint, int num_elements)
{
	for(int i=0; i< num_elements; i++)
	{
		cout<<arrayToBePrint[i]<<" ";	
	}
	cout<<"\n";
}

void initialiseArrayHost(unsigned int *d_edge_mapping, int num_edges)
{
	for(int i =0; i< num_edges; i++)
	{
		d_edge_mapping[i] = i;
	}
}

void clearArrayHost(unsigned int *array, int num_elements)
{
	for(int i=0; i< num_elements; i++)
	{
		array[i] = 0;	
	}
}


// Functions end here

void readFile(char *filename)
{
	cout<<"Now, start reading the input graph"<<endl;
	FILE *fp;
	fp = fopen(filename, "r");
	fscanf(fp,"%d",&num_vertices);
	cout<<"Number of vertices:"<<num_vertices<<endl;
	h_vertices = new unsigned int[num_vertices];
	h_length = new unsigned int[num_vertices];
	for (int i = 0; i<num_vertices; i++)
	{
		fscanf(fp,"%d %d",&h_vertices[i], &h_length[i]);
	}
	
	fscanf(fp,"%d",&source);
	cout<<"Source :"<<source<<endl;
	
	fscanf(fp,"%d",&num_edges);
	cout<<"Number of edges:"<<num_edges<<endl;
	
	h_edges = new unsigned int[num_edges];
	h_weights = new unsigned int[num_edges];
	edges_orig = new unsigned int[num_edges];
	weights_orig = new unsigned int[num_edges];
	for(int i=0; i<num_edges;i++)
	{
		fscanf(fp,"%d %d",&h_edges[i], &h_weights[i]);	
	}
	
	fclose(fp);
	
	num_vertices_orig = num_vertices;
	num_edges_orig = num_edges;
	
	//printVertices(h_vertices);
	//printEdgeWithWeights(h_edges, h_weights);
}



void alloc()
{
	size_t size_edge = num_edges*sizeof(unsigned int);
	size_t size_vertex = num_vertices*sizeof(unsigned int);
	
	config_segmented_min.algorithm = CUDPP_SEGMENTED_SCAN;
	config_segmented_min.op = CUDPP_MIN;
	config_segmented_min.datatype = CUDPP_UINT;
	config_segmented_min.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;

	config_scan_add.algorithm = CUDPP_SCAN;
	config_scan_add.op = CUDPP_ADD;
	config_scan_add.datatype = CUDPP_UINT;
	config_scan_add.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;



	//Allocate main graph variables
	CUDA_SAFE_CALL( cudaMalloc( &d_edges, size_edge));

	CUDA_SAFE_CALL( cudaMalloc( &d_vertices, size_vertex));

	CUDA_SAFE_CALL( cudaMalloc( &d_weights, size_edge));

	
	CUDA_SAFE_CALL( cudaMemcpy( d_edges, h_edges, size_edge, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL( cudaMemcpy( d_vertices, h_vertices, size_vertex, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL( cudaMemcpy( d_weights, h_weights, size_edge, cudaMemcpyHostToDevice));
	
	h_segmented_min_scan_input = new unsigned int[num_edges];
	CUDA_SAFE_CALL(cudaMalloc( &d_segmented_min_scan_input, size_edge)); 
	
	h_edge_flag = new unsigned int[num_edges];
	CUDA_SAFE_CALL(cudaMalloc( &d_edge_flag, size_edge));
	
	h_segmented_min_scan_output = new unsigned int[num_edges];
	CUDA_SAFE_CALL(cudaMalloc( &d_segmented_min_scan_output, size_edge));
	
	h_successor = new unsigned int[num_vertices];
	CUDA_SAFE_CALL(cudaMalloc( &d_successor, size_vertex));
	
	h_pick_array = new int[num_edges];
	CUDA_SAFE_CALL(cudaMalloc( &d_pick_array, sizeof(int)*num_edges));
	
	
	h_successor_copy = new unsigned int[num_vertices];
	CUDA_SAFE_CALL(cudaMalloc( &d_successor_copy, size_vertex));
	
	h_output_MST = new unsigned int[num_edges];
	CUDA_SAFE_CALL(cudaMalloc( &d_output_MST, size_edge));
	clearArrayHost(h_output_MST, num_edges);
	CUDA_SAFE_CALL( cudaMemcpy( d_output_MST, h_output_MST, size_edge, cudaMemcpyHostToDevice));
	
	
	h_append_Ids = new unsigned long long int[num_vertices];
	CUDA_SAFE_CALL(cudaMalloc( &d_append_Ids, sizeof(unsigned long long int)*num_vertices));

	h_vertex_flag = new unsigned int[num_vertices];
	CUDA_SAFE_CALL(cudaMalloc( &d_vertex_flag, size_vertex));

	h_new_vertex_Ids = new unsigned int[num_vertices];
	CUDA_SAFE_CALL(cudaMalloc( &d_new_vertex_Ids, size_vertex));

	h_old_uID = new unsigned int[num_edges];
	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_old_uID, size_edge));
	
	h_appended_uindex = new unsigned long long int[num_edges];
	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_appended_uindex, sizeof(unsigned long long int)*num_edges));
	
	
	h_edge_mapping = new unsigned int[num_edges];
	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_edge_mapping, size_edge)); 
	initialiseArrayHost(h_edge_mapping, num_edges);
	CUDA_SAFE_CALL( cudaMemcpy( d_edge_mapping, h_edge_mapping, size_edge, cudaMemcpyHostToDevice));
	
	h_edge_mapping_copy = new unsigned int[num_edges];
	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_edge_mapping_copy, size_edge)); 
	

	//Variables
	CUDA_SAFE_CALL(cudaMalloc( &d_change, sizeof(bool)));
	CUDA_SAFE_CALL(cudaMalloc( &d_new_edges, sizeof(unsigned int)));
	CUDA_SAFE_CALL(cudaMalloc( &d_new_edge_size, sizeof(unsigned int)));
	CUDA_SAFE_CALL(cudaMalloc( &d_new_vertex_size, sizeof(unsigned int)));
}


void SetGridThreadLen(unsigned int number, unsigned int *num_blocks, unsigned int *num_threads_per_block)
{
	*num_blocks = 1;
	*num_threads_per_block = number;

	
	if(number>MAX_THREADS_PER_BLOCK)
	{
		*num_blocks = (unsigned int)ceil(number/(float)MAX_THREADS_PER_BLOCK); 
		*num_threads_per_block = MAX_THREADS_PER_BLOCK; 
	}
}



void MST()
{

	unsigned int num_blocks, num_threads_per_block;
	SetGridThreadLen(num_edges, &num_blocks, &num_threads_per_block);
	dim3 grid_edgelen(num_blocks, 1, 1);
	dim3 threads_edgelen(num_threads_per_block, 1, 1);
	
	SetGridThreadLen(num_vertices, &num_blocks, &num_threads_per_block);
	dim3 grid_vertexlen(num_blocks, 1, 1);
	dim3 threads_vertexlen(num_threads_per_block, 1, 1);
	
	appendWeight<<< grid_edgelen, threads_edgelen, 0>>>(d_segmented_min_scan_input, d_weights, d_edges, num_edges);

//	cudaMemcpy( h_segmented_min_scan_input, d_segmented_min_scan_input, num_edges*sizeof(unsigned int), cudaMemcpyDeviceToHost);

//	printArrayHost(h_segmented_min_scan_input, num_edges);
	
	clearArray<<< grid_edgelen, threads_edgelen, 0>>>(d_edge_flag, num_edges);

	markEdgeFlag<<< grid_vertexlen, threads_vertexlen, 0>>>(d_edge_flag, d_vertices, num_vertices);

//	cudaMemcpy( h_edge_flag, d_edge_flag, num_edges*sizeof(unsigned int), cudaMemcpyDeviceToHost);
//	printArrayHost(h_edge_flag, num_edges);

	cudppCreate(&theCudpp);
	cudppPlan(theCudpp, &segmentedScanPlan_min, config_segmented_min, num_edges, 1, 0 ); 
	cudppSegmentedScan(segmentedScanPlan_min, d_segmented_min_scan_output, d_segmented_min_scan_input, (const unsigned int*)d_edge_flag, num_edges);
	cudppDestroyPlan(segmentedScanPlan_min);
	cudppDestroy(theCudpp);

//	segmentedMinScan(h_segmented_min_scan_output, h_segmented_min_scan_input, (const unsigned int*)h_edge_flag, num_edges);
		
	makeSuccesorArray<<<grid_vertexlen, threads_vertexlen, 0>>>(d_successor, d_vertices, d_segmented_min_scan_output, num_vertices, num_edges);
	

	removeCycles<<<grid_vertexlen, threads_vertexlen, 0>>>(d_successor, num_vertices);
	
//	cudaMemcpy( h_successor, d_successor, num_vertices*sizeof(unsigned int), cudaMemcpyDeviceToHost);
//	printArrayHost(h_successor, num_vertices);


	clearArray<<<grid_edgelen, threads_edgelen, 0>>>(d_edge_flag, num_edges);



	markFlagForUid<<<grid_vertexlen, threads_vertexlen, 0>>>(d_edge_flag, d_vertices, num_vertices);
//	cudaMemcpy( h_edge_flag,d_edge_flag, num_edges*sizeof(unsigned int), cudaMemcpyDeviceToHost);
//	printArrayHost(h_edge_flag, num_edges);

	cudppCreate(&theCudpp);
	scanPlan_add=0;
	cudppPlan(theCudpp,&scanPlan_add, config_scan_add, num_edges , 1, 0);
	cudppScan(scanPlan_add, d_old_uID, d_edge_flag, num_edges);
	cudppDestroyPlan(scanPlan_add);
//	cudaMemcpy( h_old_uID,d_old_uID, num_edges*sizeof(unsigned int), cudaMemcpyDeviceToHost);
//	printArrayHost(h_old_uID, num_edges);
	
	modifyOldUID<<<grid_edgelen, threads_edgelen, 0>>>(d_old_uID, num_edges);
//	cudaMemcpy( h_old_uID,d_old_uID, num_edges*sizeof(unsigned int), cudaMemcpyDeviceToHost);
//	printArrayHost(h_old_uID, num_edges);


//	prefixSum(h_old_uID, h_edge_flag, num_edges);
//	printArray(h_old_uID, num_edges);
	
	clearArrayInt<<<grid_edgelen, threads_edgelen, 0>>>(d_pick_array, num_edges);
	makePickArray<<<grid_edgelen, threads_edgelen, 0>>>(d_pick_array,d_successor,d_vertices,d_old_uID,num_vertices,num_edges);
//	cudaMemcpy( h_pick_array,d_pick_array, num_edges*sizeof(unsigned int), cudaMemcpyDeviceToHost);
//	printArrayHostInt(h_pick_array, num_edges);
	
	

	markOutputEdges<<<grid_edgelen, threads_edgelen, 0>>>(d_pick_array, d_segmented_min_scan_output, d_output_MST, d_edge_mapping, num_edges);
	cudaMemcpy( h_output_MST,d_output_MST, num_edges*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	printArrayHost(h_output_MST, num_edges);
	
	
//	pointer doubling code starts here 
	
	do{
		change =false;
		CUDA_SAFE_CALL(cudaMemcpy(d_change, &change, sizeof(bool), cudaMemcpyHostToDevice));
		copyArray<<<grid_vertexlen, threads_vertexlen, 0>>>(d_successor_copy, d_successor, num_vertices);
		doPointerDoubling<<<grid_vertexlen, threads_vertexlen, 0>>>(d_successor, d_successor_copy, num_vertices, d_change);
		copyArray<<<grid_vertexlen, threads_vertexlen, 0>>>(d_successor, d_successor_copy, num_vertices);
		CUDA_SAFE_CALL(cudaMemcpy(&change, d_change, sizeof(bool), cudaMemcpyDeviceToHost));

	}while(change);

//	cudaMemcpy( h_successor, d_successor, num_vertices*sizeof(unsigned int), cudaMemcpyDeviceToHost);
//	printArrayHost(h_successor, num_vertices);

	
	
	
	appendVertexId<<<grid_vertexlen, threads_vertexlen, 0>>>(d_append_Ids, d_successor, num_vertices);

//	cudaMemcpy(h_append_Ids, d_append_Ids, num_vertices*sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
//	for(int i=0; i< num_vertices; i++)
	{
//		cout<<h_append_Ids[i]<<" ";	
	}
//	cout<<"\n";

	thrust::device_ptr<unsigned long long int> sort_super_vertex(d_append_Ids);
	thrust::sort(sort_super_vertex, sort_super_vertex + num_vertices);
//	cudaMemcpy(h_append_Ids, d_append_Ids, num_vertices*sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
//	for(int i=0; i< num_vertices; i++)
	{
//		cout<<h_append_Ids[i]<<" ";	
	}
//	cout<<"\n";

//	sortSuperVertexIds(h_append_Ids, num_vertices);
//	for(int i=0; i< num_vertices; i++)
//	{
//		cout<<h_append_Ids[i]<<" ";	
//	}
//	cout<<"\n";

	
	clearArray<<<grid_vertexlen, threads_vertexlen, 0>>>(d_vertex_flag, num_vertices);

	markVertexFlag<<<grid_vertexlen, threads_vertexlen, 0>>>(d_vertex_flag, d_append_Ids, num_vertices);

//	cudaMemcpy( h_vertex_flag, d_vertex_flag, num_vertices*sizeof(unsigned int), cudaMemcpyDeviceToHost);
//	printArrayHost(h_vertex_flag, num_vertices);

	
//	prefixSum(h_new_vertex_Ids, h_vertex_flag, num_vertices);
//	printArray(h_new_vertex_Ids, num_vertices);
	
	scanPlan_add=0;
	cudppPlan(theCudpp,&scanPlan_add, config_scan_add, num_vertices , 1, 0);
	cudppScan(scanPlan_add, d_new_vertex_Ids, d_vertex_flag, num_vertices);
	cudppDestroyPlan(scanPlan_add);
//	cudaMemcpy( h_new_vertex_Ids, d_new_vertex_Ids, num_vertices*sizeof(unsigned int), cudaMemcpyDeviceToHost);
//	printArrayHost(h_new_vertex_Ids, num_vertices);
	

	markSuperVertexIdsPerVertex<<<grid_vertexlen, threads_vertexlen, 0>>>(d_new_vertex_Ids, d_append_Ids, d_vertex_flag, num_vertices);
	
//	printArray(h_vertex_flag, num_vertices);
	copyArray<<<grid_vertexlen, threads_vertexlen, 0>>>(d_new_vertex_Ids, d_vertex_flag, num_vertices);
//	cudaMemcpy( h_new_vertex_Ids, d_new_vertex_Ids, num_vertices*sizeof(unsigned int), cudaMemcpyDeviceToHost);
//	printArrayHost(h_new_vertex_Ids, num_vertices);

	
//	New vertex ids now contains the supervertex ids of the vertexes in the input order of vertices


//	printArray(h_old_uID, num_edges); //contains the source vertex id of the edges
//	printArray(h_edges, num_edges); //contains the destination vertex id of the edges

	removeSelfEdges<<<grid_edgelen, threads_edgelen, 0>>>(d_edges, d_old_uID, d_new_vertex_Ids, num_edges);
//	cudaMemcpy( h_edges, d_edges, num_edges*sizeof(unsigned int), cudaMemcpyDeviceToHost);
//	printArrayHost(h_edges, num_edges);

	//Now h_edges contains the destination vertex id of different super set or INF if it is itself.
	//Now I need to take the edges for which the supervertex ids are different
	
	
	appendForNoDuplicateEdgeRemoval<<<grid_edgelen, threads_edgelen, 0>>>(d_appended_uindex, d_edges, d_old_uID, d_weights,d_new_vertex_Ids, num_edges);
//	cudaMemcpy( h_appended_uindex, d_appended_uindex, num_edges*sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
//	for(int i=0; i< num_edges; i++)
//	{
//		cout<<h_appended_uindex[i]<<" ";	
//	}
//	cout<<"\n";

	thrust::device_ptr<unsigned long long int> sort_super_vertex_edges(d_appended_uindex);
	thrust::sort(sort_super_vertex_edges, sort_super_vertex_edges + num_edges);
//	cudaMemcpy( h_appended_uindex, d_appended_uindex, num_edges*sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
//	for(int i=0; i< num_edges; i++)
//	{
//		cout<<h_appended_uindex[i]<<" ";	
//	}
//	cout<<"\n";


//	sortArray(h_appended_uindex, num_edges);

	new_edges = (unsigned int)(num_edges+1);
	CUDA_SAFE_CALL(cudaMemcpy(d_new_edges, &new_edges, sizeof(unsigned int), cudaMemcpyHostToDevice));
	clearArray<<<grid_edgelen, threads_edgelen, 0>>>(d_edge_flag, num_edges);
	
	markEdgesNew<<<grid_edgelen, threads_edgelen, 0>>>(d_edge_flag, d_appended_uindex, d_new_edges, num_edges);
//	cudaMemcpy( h_edge_flag, d_edge_flag, num_edges*sizeof(unsigned int), cudaMemcpyDeviceToHost);
//	printArrayHost(h_edge_flag, num_edges);

	CUDA_SAFE_CALL(cudaMemcpy(&new_edges, d_new_edges, sizeof(unsigned int), cudaMemcpyDeviceToHost));
//	cout<<new_edges<<endl;


	if((new_edges == (unsigned int)(num_edges+1)) && (iteration == 0))
	{
		num_vertices = 1;
		return;
	}
	
	
	
	clearArray<<<grid_edgelen, threads_edgelen, 0>>>(d_segmented_min_scan_input, num_edges);
	clearArray<<<grid_edgelen, threads_edgelen, 0>>>(d_segmented_min_scan_output, num_edges);
	clearArrayInt<<<grid_edgelen, threads_edgelen, 0>>>(d_pick_array, num_edges);
	clearArray<<<grid_edgelen, threads_edgelen, 0>>>(d_edge_mapping_copy, num_edges);
	new_vertex_size = 0;
	new_edge_size =0;
	CUDA_SAFE_CALL(cudaMemcpy(d_new_edge_size, &new_edge_size, sizeof(unsigned int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_new_vertex_size, &new_vertex_size, sizeof(unsigned int), cudaMemcpyHostToDevice));
	
	SetGridThreadLen(new_edges, &num_blocks, &num_threads_per_block);
	dim3 grid_validsizelen(num_blocks, 1, 1);
	dim3 threads_validsizelen(num_threads_per_block, 1, 1);
	

	compactEdgeListDuplicates<<< grid_validsizelen, threads_validsizelen, 0>>>(d_edges, d_weights, d_new_vertex_Ids, d_edge_mapping, d_edge_mapping_copy, 
								d_segmented_min_scan_input, d_segmented_min_scan_output, d_edge_flag,
								d_appended_uindex, d_pick_array, d_new_edges, d_new_vertex_size, d_new_edge_size);
		
//	cudaMemcpy( h_segmented_min_scan_input, d_segmented_min_scan_input, num_edges*sizeof(unsigned int), cudaMemcpyDeviceToHost);			
//	printArrayHost(h_segmented_min_scan_input, new_edges);
//	cudaMemcpy( h_segmented_min_scan_output, d_segmented_min_scan_output, num_edges*sizeof(unsigned int), cudaMemcpyDeviceToHost);
//	printArrayHost(h_segmented_min_scan_output, new_edges);
//	cudaMemcpy( h_pick_array, d_pick_array, num_edges*sizeof(int), cudaMemcpyDeviceToHost);		
//	printArrayHostInt(h_pick_array, new_edges);
//	cudaMemcpy( h_edge_mapping_copy, d_edge_mapping_copy, num_edges*sizeof(unsigned int), cudaMemcpyDeviceToHost);		
//	printArrayHost(h_edge_mapping_copy, new_edges);
	
	

	CUDA_SAFE_CALL(cudaMemcpy(&new_edge_size, d_new_edge_size, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(&new_vertex_size, d_new_vertex_size, sizeof(unsigned int), cudaMemcpyDeviceToHost));
							
//	cout<<new_edge_size<<endl;
//	cout<<new_vertex_size<<endl;

	copyEdgeMapping<<< grid_validsizelen, threads_validsizelen, 0>>>(d_edge_mapping_copy, d_edge_mapping, d_weights, d_edges, 
					d_segmented_min_scan_input, d_segmented_min_scan_output, d_new_edges);
	
	
//	cudaMemcpy( h_edges, d_edges, new_edges*sizeof(unsigned int), cudaMemcpyDeviceToHost);
//	printArrayHost(h_edges, new_edges);
//	cudaMemcpy( h_weights, d_weights, new_edges*sizeof(unsigned int), cudaMemcpyDeviceToHost);
//	printArrayHost(h_weights, new_edges);
	clearArray<<< grid_edgelen, threads_edgelen, 0>>>(d_edge_flag, num_edges);
	clearArray<<< grid_vertexlen, threads_vertexlen, 0>>>(d_vertices, num_vertices);


//	printArrayInt(h_pick_array, new_edges);
	
	makeFlagArrayForNewVertexList<<< grid_edgelen, threads_edgelen, 0>>>(d_edge_flag, d_pick_array, num_edges);

//	cudaMemcpy( h_edge_flag, d_edge_flag, num_edges*sizeof(unsigned int), cudaMemcpyDeviceToHost);
//	printArrayHost(h_edge_flag, num_edges);

	makeVertexList<<< grid_edgelen, threads_edgelen, 0>>>(d_vertices,d_edge_flag, d_pick_array, num_edges);

//	cudaMemcpy( h_vertices, d_vertices, num_vertices*sizeof(unsigned int), cudaMemcpyDeviceToHost);
//	printArrayHost(h_vertices, num_vertices);
	
	num_edges = new_edge_size;
	num_vertices = new_vertex_size;	
//	cout<<num_edges<<endl;
//	cout<<num_vertices<<endl;


}

void freeAlloc()
{
	cout<<"Freeing the memory allocations"<<endl;
	
	
	CUDA_SAFE_CALL(cudaFree(d_edges));
	CUDA_SAFE_CALL(cudaFree(d_vertices));
	CUDA_SAFE_CALL(cudaFree(d_weights));
	CUDA_SAFE_CALL(cudaFree(d_segmented_min_scan_input));
	CUDA_SAFE_CALL(cudaFree(d_segmented_min_scan_output));
	CUDA_SAFE_CALL(cudaFree(d_edge_flag));
	CUDA_SAFE_CALL(cudaFree(d_pick_array));
	CUDA_SAFE_CALL(cudaFree(d_successor));
	CUDA_SAFE_CALL(cudaFree(d_successor_copy));
	CUDA_SAFE_CALL(cudaFree(d_output_MST));

	CUDA_SAFE_CALL(cudaFree(d_append_Ids));
	CUDA_SAFE_CALL(cudaFree(d_vertex_flag));
	CUDA_SAFE_CALL(cudaFree(d_new_vertex_Ids));
	CUDA_SAFE_CALL(cudaFree(d_old_uID));
	CUDA_SAFE_CALL(cudaFree(d_appended_uindex));

	CUDA_SAFE_CALL(cudaFree(d_edge_mapping));
	CUDA_SAFE_CALL(cudaFree(d_edge_mapping_copy));
	
	CUDA_SAFE_CALL(cudaFree(d_change));
	CUDA_SAFE_CALL(cudaFree(d_new_edges));
	CUDA_SAFE_CALL(cudaFree(d_new_edge_size));
	CUDA_SAFE_CALL(cudaFree(d_new_vertex_size));
	
	
	
	delete[] h_vertices;
	delete[] h_length;
	delete[] h_edges;
	delete[] h_weights;
	delete[] edges_orig;
	delete[] weights_orig;
	delete[] h_segmented_min_scan_input;
	delete[] h_edge_flag;
	delete[] h_segmented_min_scan_output;
	delete[] h_successor;
	delete[] h_old_uID;
	delete[] h_pick_array;
	delete[] h_output_MST;
	delete[] h_edge_mapping;
	delete[] h_successor_copy;
	delete[] h_append_Ids; //d_vertex_sort
	delete[] h_vertex_flag;
	delete[] h_new_vertex_Ids;
	delete[] h_appended_uindex;
	delete[] h_edge_mapping_copy;
}


int main(int argc, char **argv)
{
	/*(argc != 2)
		cout<<"Please give us a graph to compute"<<endl;
	*/
	time_t t0, t1;
        clock_t c0,c1;
	char *fileName = "input_100000_new";
	t0=time(NULL);
        c0=clock();

        printf ("\tbegin (wall):            %ld\n", (long) t0);
        printf ("\tbegin (CPU):             %d\n", (int) c0);
	readFile(fileName);
	
	t1=time(NULL);
        c1=clock();
        printf ("\telapsed wall clock time: %ld\n", (long) (t1 - t0));
        printf ("\telapsed CPU time:        %f\n", (float) (c1 - c0)/CLOCKS_PER_SEC);

	for(int i=0; i< num_edges_orig;i++)
	{
		edges_orig[i]=h_edges[i];
		weights_orig[i]=h_weights[i];
	}
	alloc();
	do{
		
		MST();
		iteration= iteration+1;
	}while(num_vertices > 1);
	
	CUDA_SAFE_CALL(cudaMemcpy( h_output_MST,d_output_MST, num_edges_orig*sizeof(unsigned int), cudaMemcpyDeviceToHost));
	printArrayHost(h_output_MST, num_edges_orig);	

	for(int i=0; i<num_edges_orig; i++)
	{
		if(h_output_MST[i]==1)
		{
			cout<<"Edge ="<<edges_orig[i]<<" Weight = "<<weights_orig[i]<<endl;
			MSTWeight = MSTWeight+weights_orig[i];
		}
	}
	cout<<"Weight of MST is : "<<MSTWeight<<endl;

	freeAlloc();
	return 0;	
}

