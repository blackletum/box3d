// SPDX-FileCopyrightText: 2026 Erin Catto
// SPDX-License-Identifier: MIT

#include "shape.h"
#include "test_macros.h"

#include "box3d/box3d.h"
#include "box3d/collision.h"
#include "box3d/math_functions.h"

#include <float.h>
#include <string.h>

// Two quads meeting at a concave crease along the shared edge 1-4. The crease exercises
// edge identification, which reads the baked winding rather than the input winding.
#define VALLEY_VERTEX_COUNT 6
#define VALLEY_TRIANGLE_COUNT 4

static void MakeValley( b3Vec3* vertices, int32_t* indices )
{
	b3Vec3 v[VALLEY_VERTEX_COUNT] = {
		{ -1.0f, 1.0f, -1.0f }, { 0.0f, 0.0f, -1.0f }, { 1.0f, 1.0f, -1.0f },
		{ -1.0f, 1.0f, 1.0f },	{ 0.0f, 0.0f, 1.0f },  { 1.0f, 1.0f, 1.0f },
	};

	int32_t i[3 * VALLEY_TRIANGLE_COUNT] = {
		0, 3, 1, // left quad
		3, 4, 1,
		1, 4, 2, // right quad
		4, 5, 2,
	};

	memcpy( vertices, v, sizeof( v ) );
	memcpy( indices, i, sizeof( i ) );
}

// A vertex laid out the way a renderer would hand it over. The position sits off the
// front of the struct so a non-zero base offset gets exercised too.
typedef struct FatVertex
{
	float weight;
	b3Vec3 position;
	float uv[2];
} FatVertex;

static void MakeFatVertices( FatVertex* fat, const b3Vec3* vertices, int count )
{
	for ( int i = 0; i < count; ++i )
	{
		// Poison the padding so a stride mistake bakes obvious garbage instead of near misses
		fat[i].weight = 1000.0f + (float)i;
		fat[i].position = vertices[i];
		fat[i].uv[0] = -1000.0f;
		fat[i].uv[1] = -2000.0f;
	}
}

static void ReverseWinding( int32_t* indices, int triangleCount )
{
	for ( int i = 0; i < triangleCount; ++i )
	{
		int32_t temp = indices[3 * i + 1];
		indices[3 * i + 1] = indices[3 * i + 2];
		indices[3 * i + 2] = temp;
	}
}

// The hash covers every byte of the mesh block, so this compares the tree, the vertices,
// the baked triangles and the edge flags in one shot.
static bool MeshesMatch( const b3MeshData* mesh1, const b3MeshData* mesh2 )
{
	return mesh1->byteCount == mesh2->byteCount && mesh1->hash == mesh2->hash;
}

static bool VerticesMatch( const b3MeshData* mesh, const b3Vec3* expected, int count )
{
	if ( mesh->vertexCount != count )
	{
		return false;
	}

	const b3Vec3* vertices = b3GetMeshVertices( mesh );
	for ( int i = 0; i < count; ++i )
	{
		if ( b3DistanceSquared( vertices[i], expected[i] ) > FLT_EPSILON )
		{
			return false;
		}
	}

	return true;
}

// The valley faces up, so every baked triangle should wind counter clockwise about +Y
static bool FacesUp( const b3MeshData* mesh )
{
	const b3Vec3* vertices = b3GetMeshVertices( mesh );
	const b3MeshTriangle* triangles = b3GetMeshTriangles( mesh );

	for ( int i = 0; i < mesh->triangleCount; ++i )
	{
		b3Vec3 v1 = vertices[triangles[i].index1];
		b3Vec3 v2 = vertices[triangles[i].index2];
		b3Vec3 v3 = vertices[triangles[i].index3];

		b3Vec3 normal = b3Cross( b3Sub( v2, v1 ), b3Sub( v3, v1 ) );
		if ( normal.y <= 0.0f )
		{
			return false;
		}
	}

	return true;
}

static bool HasConcaveEdge( const b3MeshData* mesh )
{
	const uint8_t* flags = b3GetMeshFlags( mesh );

	for ( int i = 0; i < mesh->triangleCount; ++i )
	{
		if ( ( flags[i] & b3_allConcaveEdges ) != 0 )
		{
			return true;
		}
	}

	return false;
}

static b3MeshDef MakeValleyDef( b3Vec3* vertices, int32_t* indices )
{
	b3MeshDef def = { 0 };
	def.vertices = vertices;
	def.stride = sizeof( b3Vec3 );
	def.indices = indices;
	def.vertexCount = VALLEY_VERTEX_COUNT;
	def.triangleCount = VALLEY_TRIANGLE_COUNT;
	def.identifyEdges = true;
	return def;
}

// Dense input is the reference the other cases are measured against
static int MeshDenseStride( void )
{
	b3Vec3 vertices[VALLEY_VERTEX_COUNT];
	int32_t indices[3 * VALLEY_TRIANGLE_COUNT];
	MakeValley( vertices, indices );

	b3MeshDef def = MakeValleyDef( vertices, indices );
	b3MeshData* mesh = b3CreateMesh( &def, NULL, 0 );
	ENSURE( mesh != NULL );

	ENSURE( mesh->vertexCount == VALLEY_VERTEX_COUNT );
	ENSURE( mesh->triangleCount == VALLEY_TRIANGLE_COUNT );
	ENSURE( mesh->degenerateCount == 0 );
	ENSURE( VerticesMatch( mesh, vertices, VALLEY_VERTEX_COUNT ) );
	ENSURE( FacesUp( mesh ) );
	ENSURE( HasConcaveEdge( mesh ) );

	b3DestroyMesh( mesh );
	return 0;
}

// Interleaved input must bake to exactly the same mesh as dense input
static int MeshFatStride( void )
{
	b3Vec3 vertices[VALLEY_VERTEX_COUNT];
	int32_t indices[3 * VALLEY_TRIANGLE_COUNT];
	MakeValley( vertices, indices );

	b3MeshDef denseDef = MakeValleyDef( vertices, indices );
	b3MeshData* denseMesh = b3CreateMesh( &denseDef, NULL, 0 );
	ENSURE( denseMesh != NULL );

	FatVertex fat[VALLEY_VERTEX_COUNT];
	MakeFatVertices( fat, vertices, VALLEY_VERTEX_COUNT );

	b3MeshDef fatDef = MakeValleyDef( &fat[0].position, indices );
	fatDef.stride = sizeof( FatVertex );
	b3MeshData* fatMesh = b3CreateMesh( &fatDef, NULL, 0 );
	ENSURE( fatMesh != NULL );

	ENSURE( VerticesMatch( fatMesh, vertices, VALLEY_VERTEX_COUNT ) );
	ENSURE( FacesUp( fatMesh ) );
	ENSURE( MeshesMatch( denseMesh, fatMesh ) );

	b3DestroyMesh( fatMesh );
	b3DestroyMesh( denseMesh );
	return 0;
}

// Welding reads the source vertices through the stride as well
static int MeshStrideWeld( void )
{
	b3Vec3 vertices[VALLEY_VERTEX_COUNT];
	int32_t indices[3 * VALLEY_TRIANGLE_COUNT];
	MakeValley( vertices, indices );

	// Split the crease so each quad owns a copy of the shared edge
	b3Vec3 splitVertices[8];
	memcpy( splitVertices, vertices, sizeof( vertices ) );
	splitVertices[6] = vertices[1];
	splitVertices[7] = vertices[4];

	// Point the right quad at the copies
	int32_t splitIndices[3 * VALLEY_TRIANGLE_COUNT];
	memcpy( splitIndices, indices, sizeof( indices ) );
	splitIndices[6] = 6;
	splitIndices[7] = 7;
	splitIndices[9] = 7;

	FatVertex fat[8];
	MakeFatVertices( fat, splitVertices, 8 );

	b3MeshDef def = MakeValleyDef( &fat[0].position, splitIndices );
	def.stride = sizeof( FatVertex );
	def.vertexCount = 8;
	def.weldVertices = true;
	def.weldTolerance = 0.01f;

	b3MeshData* mesh = b3CreateMesh( &def, NULL, 0 );
	ENSURE( mesh != NULL );

	ENSURE( mesh->vertexCount == VALLEY_VERTEX_COUNT );
	ENSURE( mesh->triangleCount == VALLEY_TRIANGLE_COUNT );
	ENSURE( VerticesMatch( mesh, vertices, VALLEY_VERTEX_COUNT ) );
	ENSURE( FacesUp( mesh ) );
	ENSURE( HasConcaveEdge( mesh ) );

	b3DestroyMesh( mesh );
	return 0;
}

// Clockwise input plus the flag must bake to the same mesh as counter clockwise input
static int MeshClockWise( void )
{
	b3Vec3 vertices[VALLEY_VERTEX_COUNT];
	int32_t indices[3 * VALLEY_TRIANGLE_COUNT];
	MakeValley( vertices, indices );

	b3MeshDef ccwDef = MakeValleyDef( vertices, indices );
	b3MeshData* ccwMesh = b3CreateMesh( &ccwDef, NULL, 0 );
	ENSURE( ccwMesh != NULL );

	int32_t reversed[3 * VALLEY_TRIANGLE_COUNT];
	memcpy( reversed, indices, sizeof( indices ) );
	ReverseWinding( reversed, VALLEY_TRIANGLE_COUNT );

	b3MeshDef cwDef = MakeValleyDef( vertices, reversed );
	cwDef.clockWiseWinding = true;
	b3MeshData* cwMesh = b3CreateMesh( &cwDef, NULL, 0 );
	ENSURE( cwMesh != NULL );

	ENSURE( FacesUp( cwMesh ) );
	ENSURE( HasConcaveEdge( cwMesh ) );
	ENSURE( MeshesMatch( ccwMesh, cwMesh ) );

	b3DestroyMesh( cwMesh );
	b3DestroyMesh( ccwMesh );
	return 0;
}

// Without the flag the same clockwise input must bake inside out
static int MeshClockWiseIgnored( void )
{
	b3Vec3 vertices[VALLEY_VERTEX_COUNT];
	int32_t indices[3 * VALLEY_TRIANGLE_COUNT];
	MakeValley( vertices, indices );
	ReverseWinding( indices, VALLEY_TRIANGLE_COUNT );

	b3MeshDef def = MakeValleyDef( vertices, indices );
	b3MeshData* mesh = b3CreateMesh( &def, NULL, 0 );
	ENSURE( mesh != NULL );

	ENSURE( FacesUp( mesh ) == false );

	b3DestroyMesh( mesh );
	return 0;
}

// Winding, stride and welding have to compose
static int MeshClockWiseStrideWeld( void )
{
	b3Vec3 vertices[VALLEY_VERTEX_COUNT];
	int32_t indices[3 * VALLEY_TRIANGLE_COUNT];
	MakeValley( vertices, indices );

	b3MeshDef ccwDef = MakeValleyDef( vertices, indices );
	ccwDef.weldVertices = true;
	ccwDef.weldTolerance = 0.01f;
	b3MeshData* ccwMesh = b3CreateMesh( &ccwDef, NULL, 0 );
	ENSURE( ccwMesh != NULL );

	int32_t reversed[3 * VALLEY_TRIANGLE_COUNT];
	memcpy( reversed, indices, sizeof( indices ) );
	ReverseWinding( reversed, VALLEY_TRIANGLE_COUNT );

	FatVertex fat[VALLEY_VERTEX_COUNT];
	MakeFatVertices( fat, vertices, VALLEY_VERTEX_COUNT );

	b3MeshDef cwDef = MakeValleyDef( &fat[0].position, reversed );
	cwDef.stride = sizeof( FatVertex );
	cwDef.clockWiseWinding = true;
	cwDef.weldVertices = true;
	cwDef.weldTolerance = 0.01f;
	b3MeshData* cwMesh = b3CreateMesh( &cwDef, NULL, 0 );
	ENSURE( cwMesh != NULL );

	ENSURE( FacesUp( cwMesh ) );
	ENSURE( MeshesMatch( ccwMesh, cwMesh ) );

	b3DestroyMesh( cwMesh );
	b3DestroyMesh( ccwMesh );
	return 0;
}

// The built in creators fill their own def, so a missed stride shows up here
static int MeshCreators( void )
{
	b3Vec3 center = { 1.0f, 2.0f, 3.0f };
	b3Vec3 extent = { 0.5f, 1.0f, 1.5f };

	b3MeshData* meshes[] = {
		b3CreateGridMesh( 4, 4, 1.0f, 2, true ),
		b3CreateWaveMesh( 4, 4, 1.0f, 0.5f, 1.0f, 1.0f ),
		b3CreateTorusMesh( 8, 6, 2.0f, 0.5f ),
		b3CreateBoxMesh( center, extent, true ),
		b3CreateHollowBoxMesh( center, extent ),
		b3CreatePlatformMesh( center, 2.0f, 1.0f, 2.0f ),
	};

	for ( int i = 0; i < ARRAY_COUNT( meshes ); ++i )
	{
		ENSURE( meshes[i] != NULL );
		ENSURE( meshes[i]->vertexCount >= 3 );
		ENSURE( meshes[i]->triangleCount >= 1 );
		ENSURE( b3IsSaneAABB( meshes[i]->bounds ) );

		b3DestroyMesh( meshes[i] );
	}

	return 0;
}

// Two bits per edge: concave and inverse concave. Both set means flat.
static int EdgeClass( int flags, int edgeIndex )
{
	int concave = ( flags >> edgeIndex ) & 1;
	int inverse = ( flags >> ( edgeIndex + 4 ) ) & 1;
	return concave | inverse << 1;
}

// A mirror with the winding swapped keeps every edge convex, concave or flat, so each edge
// class must follow its vertex pair to its new slot
static bool MirroredFlagsMatch( const b3MeshData* data, b3Vec3 mirrorScale )
{
	b3Mesh plain = { .data = data, .scale = b3Vec3_one };
	b3Mesh mirrored = { .data = data, .scale = mirrorScale };

	for ( int i = 0; i < data->triangleCount; ++i )
	{
		b3Triangle triangle1 = b3GetMeshTriangle( &plain, i );
		b3Triangle triangle2 = b3GetMeshTriangle( &mirrored, i );
		int indices1[3] = { triangle1.i1, triangle1.i2, triangle1.i3 };
		int indices2[3] = { triangle2.i1, triangle2.i2, triangle2.i3 };

		for ( int k = 0; k < 3; ++k )
		{
			int a = indices2[k];
			int b = indices2[( k + 1 ) % 3];

			int match = -1;
			for ( int j = 0; j < 3; ++j )
			{
				int c = indices1[j];
				int d = indices1[( j + 1 ) % 3];
				if ( ( a == c && b == d ) || ( a == d && b == c ) )
				{
					match = j;
				}
			}

			if ( match == -1 || EdgeClass( triangle2.flags, k ) != EdgeClass( triangle1.flags, match ) )
			{
				return false;
			}
		}
	}

	return true;
}

static bool HasConvexEdge( const b3MeshData* mesh )
{
	const uint8_t* flags = b3GetMeshFlags( mesh );

	for ( int i = 0; i < mesh->triangleCount; ++i )
	{
		for ( int k = 0; k < 3; ++k )
		{
			if ( EdgeClass( flags[i], k ) == 2 )
			{
				return true;
			}
		}
	}

	return false;
}

// The valley has concave and flat edges, the box has convex and flat edges
static int MeshMirroredFlags( void )
{
	b3Vec3 vertices[VALLEY_VERTEX_COUNT];
	int32_t indices[3 * VALLEY_TRIANGLE_COUNT];
	MakeValley( vertices, indices );

	b3MeshDef def = MakeValleyDef( vertices, indices );
	b3MeshData* valley = b3CreateMesh( &def, NULL, 0 );
	ENSURE( valley != NULL );
	ENSURE( HasConcaveEdge( valley ) );

	b3MeshData* box = b3CreateBoxMesh( b3Vec3_zero, (b3Vec3){ 1.0f, 2.0f, 3.0f }, true );
	ENSURE( box != NULL );
	ENSURE( HasConvexEdge( box ) );

	b3Vec3 mirrorScales[] = {
		{ -1.0f, 1.0f, 1.0f },
		{ 1.0f, -1.0f, 1.0f },
		{ 2.0f, 0.5f, -3.0f },
		{ -1.0f, -1.0f, -1.0f },
	};

	for ( int i = 0; i < ARRAY_COUNT( mirrorScales ); ++i )
	{
		ENSURE( MirroredFlagsMatch( valley, mirrorScales[i] ) );
		ENSURE( MirroredFlagsMatch( box, mirrorScales[i] ) );
	}

	b3DestroyMesh( box );
	b3DestroyMesh( valley );
	return 0;
}

// A wide box sliding over a fine grid sits on hundreds of flat internal edges. With no friction
// any manifold off the face normal is a ghost that slows or lifts the box.
static int WideHullSlide( b3Vec3 meshScale )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	b3WorldId worldId = b3CreateWorld( &worldDef );

	b3MeshData* mesh = b3CreateGridMesh( 120, 40, 0.25f, 0, true );
	ENSURE( mesh != NULL );

	b3BodyDef bodyDef = b3DefaultBodyDef();
	b3BodyId groundId = b3CreateBody( worldId, &bodyDef );
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	b3CreateMeshShape( groundId, &shapeDef, mesh, meshScale );

	float speed = 4.0f;
	bodyDef.type = b3_dynamicBody;
	bodyDef.position = (b3Pos){ -10.0f, 0.5f, 0.0f };
	bodyDef.linearVelocity = (b3Vec3){ speed, 0.0f, 0.0f };
	b3BodyId boxId = b3CreateBody( worldId, &bodyDef );

	b3BoxHull box = b3MakeBoxHull( 1.25f, 0.5f, 1.25f );
	shapeDef.baseMaterial.friction = 0.0f;
	b3CreateHullShape( boxId, &shapeDef, &box.base );

	int ghostCount = 0;
	for ( int step = 0; step < 120; ++step )
	{
		b3World_Step( worldId, 1.0f / 60.0f, 4 );

		b3ContactData contacts[4];
		int contactCount = b3Body_GetContactData( boxId, contacts, ARRAY_COUNT( contacts ) );
		for ( int i = 0; i < contactCount; ++i )
		{
			for ( int j = 0; j < contacts[i].manifoldCount; ++j )
			{
				if ( b3AbsFloat( contacts[i].manifolds[j].normal.y ) < 0.99f )
				{
					ghostCount += 1;
				}
			}
		}
	}

	b3Vec3 velocity = b3Body_GetLinearVelocity( boxId );

	ENSURE( ghostCount == 0 );
	ENSURE( velocity.x > 0.99f * speed );
	ENSURE( b3AbsFloat( velocity.y ) < 0.01f );

	b3DestroyWorld( worldId );
	b3DestroyMesh( mesh );
	return 0;
}

// The box overlaps more than a thousand grid triangles. All of them must reach the narrow
// phase, so the reduced manifold reaches all four sides of the footprint.
static int MeshManyTriangles( void )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	b3WorldId worldId = b3CreateWorld( &worldDef );

	b3MeshData* mesh = b3CreateGridMesh( 40, 40, 0.25f, 0, true );
	ENSURE( mesh != NULL );

	b3BodyDef bodyDef = b3DefaultBodyDef();
	b3BodyId groundId = b3CreateBody( worldId, &bodyDef );
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	b3CreateMeshShape( groundId, &shapeDef, mesh, b3Vec3_one );

	bodyDef.type = b3_dynamicBody;
	bodyDef.position = (b3Pos){ 0.0f, 0.5f, 0.0f };
	b3BodyId boxId = b3CreateBody( worldId, &bodyDef );

	b3BoxHull box = b3MakeBoxHull( 3.0f, 0.5f, 3.0f );
	b3ShapeId boxShapeId = b3CreateHullShape( boxId, &shapeDef, &box.base );

	for ( int step = 0; step < 30; ++step )
	{
		b3World_Step( worldId, 1.0f / 60.0f, 4 );
	}

	b3Pos position = b3Body_GetPosition( boxId );
	b3Quat rotation = b3Body_GetRotation( boxId );

	ENSURE( b3AbsFloat( (float)position.y - 0.5f ) < 0.02f );
	ENSURE( b3AbsFloat( rotation.v.x ) < 0.001f );
	ENSURE( b3AbsFloat( rotation.v.z ) < 0.001f );

	b3Vec3 lower = { FLT_MAX, FLT_MAX, FLT_MAX };
	b3Vec3 upper = { -FLT_MAX, -FLT_MAX, -FLT_MAX };

	b3ContactData contacts[4];
	int contactCount = b3Body_GetContactData( boxId, contacts, ARRAY_COUNT( contacts ) );
	for ( int i = 0; i < contactCount; ++i )
	{
		bool boxIsA = B3_ID_EQUALS( contacts[i].shapeIdA, boxShapeId );
		for ( int j = 0; j < contacts[i].manifoldCount; ++j )
		{
			const b3Manifold* manifold = contacts[i].manifolds + j;
			for ( int k = 0; k < manifold->pointCount; ++k )
			{
				b3Vec3 anchor = boxIsA ? manifold->points[k].anchorA : manifold->points[k].anchorB;
				lower = b3Min( lower, anchor );
				upper = b3Max( upper, anchor );
			}
		}
	}

	ENSURE( lower.x < -2.5f && upper.x > 2.5f );
	ENSURE( lower.z < -2.5f && upper.z > 2.5f );

	b3DestroyWorld( worldId );
	b3DestroyMesh( mesh );
	return 0;
}

// The grid is symmetric in x, so the mirror only changes the winding and the flags
static int MeshWideHullSlide( void )
{
	ENSURE( WideHullSlide( b3Vec3_one ) == 0 );
	ENSURE( WideHullSlide( (b3Vec3){ -1.0f, 1.0f, 1.0f } ) == 0 );
	return 0;
}

int MeshTest( void )
{
	RUN_SUBTEST( MeshDenseStride );
	RUN_SUBTEST( MeshFatStride );
	RUN_SUBTEST( MeshStrideWeld );
	RUN_SUBTEST( MeshClockWise );
	RUN_SUBTEST( MeshClockWiseIgnored );
	RUN_SUBTEST( MeshClockWiseStrideWeld );
	RUN_SUBTEST( MeshCreators );
	RUN_SUBTEST( MeshWideHullSlide );
	RUN_SUBTEST( MeshManyTriangles );
	RUN_SUBTEST( MeshMirroredFlags );

	return 0;
}
