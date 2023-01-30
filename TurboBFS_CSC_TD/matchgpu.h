/*
Copyright 2011, Bas Fagginger Auer.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef MATCH_MATCH_GPU_H
#define MATCH_MATCH_GPU_H

#define NR_MATCH_ROUNDS 20
//#define NR_MAX_MATCH_ROUNDS 256
#define NR_MAX_MATCH_ROUNDS 10

__global__ void gSelect(int *match, int *dkeepMatching, const int nrVertices, const uint random);
__global__ void gaSelect(int *match, int *dkeepMatching, const int nrVertices, const uint random);
__global__ void gMatch(int *match, const int *requests, const int nrVertices);

//==== Random greedy matching kernels ====
__global__ void grRequest(int *CP_d,int *IC_d,int *requests, const int *match, const int nrVertices);
__global__ void grRespond(int *CP_d,int *IC_d,int *requests, const int *match, const int nrVertices);

//==== Weighted greedy matching kernels ====
__global__ void gwRequest(int *requests, const int *match, const int nrVertices);
__global__ void gwRespond(int *requests, const int *match, const int nrVertices);

#endif
