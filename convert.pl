#c an undirected weighted connected graph with 15 vertices and 50 edges
#p sp 15 50
#a 1 5 117
#a 1 11 603
#a 1 4 56
#a 1 13 339
#a 1 7 255
#a 2 15 414
#a 2 7 187
#a 2 3 416
#a 2 4 567
#a 2 6 851

#!/usr/bin/perl -w

use strict;
use warnings;
use IO::File;

my @trace;
my $i;
my $index;
my $prev_vertex;
my $edges;
my $start;
my @row;
my $edges_string;
my $length_of_trace;
my %edge_hash;
my $key;
my $value;
my $num_vertices;
my $num_edges;

open INPUT, $ARGV[0];
@trace = <INPUT>;
close INPUT;

chomp(@trace);

$length_of_trace = $#trace;

@row = split(' ', $trace[1]);
$num_vertices = $row[2];
$num_edges = $row[3];

print $num_vertices."\n";
for($i = 2; $i <= $length_of_trace; $i++) 
{
	@row = split(' ', $trace[$i]);

	if($i == 2) {
		$prev_vertex = $row[1];
		$edges = 0;
		$index = 0;
		$start = $i;
		$edges_string = "";
	}

	if($prev_vertex != $row[1]) {
		for $key (sort { $a <=> $b } keys %edge_hash ) {
		#	print "\n".$key."\t".$edge_hash{$key}."\n";
			$edges_string = $edges_string.$key."\t".$edge_hash{$key}."\n";
		}
		%edge_hash=();

		print $index."\t".$edges."\n";
		$index = $edges + $index;
		$edges = 1;
		$edge_hash{$row[2]-1} = $row[3];
		$prev_vertex = $row[1];
		$start = $i;
	}
	else {
		$edges = $edges+1;
		$edge_hash{$row[2]-1} = $row[3];
	}
}
	
for $key (sort { $a <=> $b } keys %edge_hash ) {
	$edges_string = $edges_string.$key."\t".$edge_hash{$key}."\n";
}

print $index."\t".$edges."\n";
print "\n0\n\n".$num_edges."\n";
print $edges_string."\n";
