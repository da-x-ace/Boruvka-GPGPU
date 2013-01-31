/*

Given n and m, generates a random connected undirected graph with m edges and 
f( n, m ) vertices, where f( n, m ) >= n - n * ( 1 - 1 / n ) ^ ( m / 2 + 1 ) 
w.h.p. in m. The edges are weighted with random integers in [ 1, MAX_W ]
( default value of MAX_W is 1000 ).

Method: We start with a graph containing n isolated vertices ( i.e., no edges ) 
among which one vertex is chosen and placed in a set S. Then in each of the next 
m / 2 iterations we add a pair of edges between two distinct random vertices 
u and v, where u and v are chosen uniformly at random from S and the entire 
graph, respectively, such that ( u, v ) and ( v, u ) do not already exist. We 
add v to set S. After adding all edges we eliminate all solated vertices and 
renumber the vertices in S from 1 to |S|. We output the graph containing the 
vertices in S along with the m edges.

*/

#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>

using namespace std;

// maximum edge weight
#define MAX_W 1000

typedef struct
	{
	 int y;                   // other end of the edge
         int w;                   // edge weight
	} EDGE_TYPE;

typedef struct
	{
	 int pos;                 // temporary storage
         int ne;                  // number of edges
         vector < EDGE_TYPE > E;  // list of edges
	} VERTEX_TYPE;

vector < VERTEX_TYPE > V;         // list of vertices

int n, m, fn;                     // n = upper bound on #vertices, m = #edges, fn = actual #vertices


vector < int > tmpv;              // temporary vector



// generate a random connected undirected graph with m edges and 
// f( n, m ) vertices, where f( n, m ) = n - n * ( 1 - 1 / n ) ^ ( m / 2 + 1 ) 
// w.h.p. in m. The edges are weighted with random integers in [ 1, MAX_W ].

void gen_graph( void )
{
   int x, y, k, l;
   EDGE_TYPE e;

   V.resize( n + 1 );

   for ( k = 1; k <= n; k++ )
     {
      V[ k ].pos = 0;
      V[ k ].ne = 0;
     }

   srand48( time( NULL ) );

   // create a trivial connected component containing only vertex 1
   V[ 1 ].pos = 1;
   tmpv.resize( 0 );
   tmpv.push_back( 1 );

   k = 0;

   while ( k < m )                                  // add m edges
     {
      x = tmpv[ lrand48( ) % tmpv.size( ) ];        // choose a random vertex from the current connected component
      y = 1 + ( lrand48( ) % n );                   // choose a random vertex from the entire graph

      if ( x != y )                                 // if not a self loop
        {
         for ( l = 0; l < V[ x ].ne; l++ )          // check whether ( x, y ) already exists
            if ( V[ x ].E[ l ].y == y ) break;

         if ( l == V[ x ].ne )                      // ( x, y ) is a new edge
            {
             e.w = 1 + ( lrand48( ) % MAX_W );      // choose a random edge weight

             e.y = y;
             V[ x ].E.push_back( e );               // add edge ( x, y )
             V[ x ].ne++;

             if ( V[ y ].pos == 0 )                 // if y was originally not in the connected component
                {
                 V[ y ].pos = 1;
                 tmpv.push_back( y );               // add y to the connected component
                }

	     e.y = x;
             V[ y ].E.push_back( e );               // add edge ( y, x )
             V[ y ].ne++;

             k += 2;                                // two edges added
            }
        }
     }

   fn = 0;

   for ( x = 1; x <= n; x++ )
      if ( V[ x ].ne > 0 ) V[ x ].pos = ++fn;      // count and renumber the vertices in the connected component
}


// print the graph in standard DIMACS Challenge 9 format

void print_graph( void )
{
   int i, j;

   printf( "c an undirected weighted connected graph with %d vertices and %d edges\n", fn, m );
   printf( "p sp %d %d\n", fn, m );

   for ( i = 1; i <= n; i++ )
      for ( j = 0; j < V[ i ].ne; j++ )
         printf( "a %d %d %d\n", V[ i ].pos, V[ V[ i ].E[ j ].y ].pos, V[ i ].E[ j ].w );
}


int main( int argc, char *argv[ ] )
{
   double mm;

   if ( ( argc < 3 ) || ( sscanf( argv[ 1 ], "%d", &n ) < 1 ) || ( sscanf( argv[ 2 ], "%d", &m ) < 1 ) )
      {
       printf( "\nUsage: %s #vertices #edges\n\n", argv[ 0 ] );
       return 1;
      }

   if ( ( n < 1 ) || ( n > 1000000000 ) || ( m < 1 ) || ( m > 1000000000 ) )
      {
       printf( "\nError: Both #vertices and #edges must be integers in [ 1, 10^9 ].\n\n", argv[ 0 ] );
       return 1;
      }

   if ( m & 1 )                     // each undirected edge counts twice ( as two directed edges )
      {
       printf( "\nError: #edges must be even.\n\n", argv[ 0 ] );
       return 1;
      }

   mm = ( 1.0 * n ) * ( n - 1 );

   if ( m > mm )                    // an undirected graph on n vertices cannot contain more than n * ( n - 1 ) edges
      {
       printf( "\nError: Too many edges!\n\n", argv[ 0 ] );
       return 1;
      }

   gen_graph( );                    // generate the graph
   print_graph( );                  // print the graph

   return 0;
}
