// SPDX-FileCopyrightText: 2026 Erin Catto
// SPDX-License-Identifier: MIT

// Headless soak of the Continuous / Mesh Drop sample. Each generation drops 1024 small bodies on a
// wave mesh and steps until every body sleeps. A generation fails if a body falls through the mesh
// or the pile is still awake at the step limit. Generations run in parallel, one single threaded
// world per thread. The printed seed reproduces a generation in the sample.

#include "core.h"

#include "box3d/box3d.h"
#include "box3d/collision.h"
#include "box3d/math_functions.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GRID_COUNT 32
#define BODY_COUNT ( GRID_COUNT * GRID_COUNT )
#define MAX_REPORTED_FAILURES 32
#define MAX_THREADS 64

typedef enum ShapeType
{
	e_box,
	e_capsule,
	e_cylinder,
	e_sphere,
	e_shapeCount,
} ShapeType;

static const char* s_shapeNames[e_shapeCount] = { "box", "capsule", "cylinder", "sphere" };

typedef enum FailureKind
{
	e_tunnel,
	e_awake,
} FailureKind;

typedef struct Failure
{
	FailureKind kind;
	ShapeType shapeType;
	int generation;
	uint32_t seed;
	int step;
	int index1;
	int index2;
	b3Pos position;
} Failure;

typedef struct Soak
{
	b3Mutex* mutex;
	b3MeshData* groundMesh;
	b3HullData* cylinder;
	uint32_t baseSeed;
	int generationCount;
	int nextGeneration;
	int shapeType;
	bool fixedSeed;

	int completedCount;
	int failureCount;
	int tunnelCount;
	int awakeCount;
	long long totalSteps;
	int maxSteps;
	Failure failures[MAX_REPORTED_FAILURES];
} Soak;

// Same XorShift32 and mapping as the samples, so a seed replays there
static float RandomFloatRange( uint32_t* seed, float lo, float hi )
{
	uint32_t x = *seed;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	*seed = x;

	int i = (int)( x % 32768 );
	float r = (float)( i & 32767 ) / 32767.0f;
	return ( hi - lo ) * r + lo;
}

static b3Vec3 RandomVec3Uniform( uint32_t* seed, float lo, float hi )
{
	b3Vec3 v;
	v.x = RandomFloatRange( seed, lo, hi );
	v.y = RandomFloatRange( seed, lo, hi );
	v.z = RandomFloatRange( seed, lo, hi );
	return v;
}

static uint32_t HashSeed( uint32_t x )
{
	x ^= x >> 16;
	x *= 0x7feb352du;
	x ^= x >> 15;
	x *= 0x846ca68bu;
	x ^= x >> 16;
	return x != 0 ? x : 1;
}

static void CreateGround( b3WorldId worldId, const b3MeshData* mesh )
{
	b3BodyDef bodyDef = b3DefaultBodyDef();
	b3BodyId groundId = b3CreateBody( worldId, &bodyDef );

	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.filter.categoryBits = 1;
	b3CreateMeshShape( groundId, &shapeDef, mesh, b3Vec3_one );

	float extent = 20.0f;
	float halfHeight = 1.0f;

	b3Transform transforms[4] = {
		{ { 0.0f, halfHeight, -extent }, b3Quat_identity },
		{ { 0.0f, halfHeight, extent }, b3Quat_identity },
		{ { -extent, halfHeight, 0.0f }, b3Quat_identity },
		{ { extent, halfHeight, 0.0f }, b3Quat_identity },
	};

	for ( int i = 0; i < 4; ++i )
	{
		float hx = i < 2 ? extent : 0.1f;
		float hz = i < 2 ? 0.1f : extent;
		b3BoxHull wallBox = b3MakeTransformedBoxHull( hx, halfHeight, hz, transforms[i] );
		b3CreateHullShape( groundId, &shapeDef, &wallBox.base );
	}
}

static void RecordFailure( Soak* soak, const Failure* failure )
{
	b3LockMutex( soak->mutex );
	if ( soak->failureCount < MAX_REPORTED_FAILURES )
	{
		soak->failures[soak->failureCount] = *failure;
	}
	soak->failureCount += 1;
	if ( failure->kind == e_tunnel )
	{
		soak->tunnelCount += 1;
	}
	else
	{
		soak->awakeCount += 1;
	}
	b3UnlockMutex( soak->mutex );

	const char* kind = failure->kind == e_tunnel ? "TUNNEL" : "AWAKE";
	printf( "%s: generation %d, %s, seed %u, step %d, body (%d, %d) at (%.2f, %.2f, %.2f)\n", kind, failure->generation,
			s_shapeNames[failure->shapeType], failure->seed, failure->step, failure->index1, failure->index2,
			(float)failure->position.x, (float)failure->position.y, (float)failure->position.z );
}

// Mirrors MeshDrop::Generate and the auto generate step loop
static int RunGeneration( Soak* soak, int generation, uint32_t seed, ShapeType shapeType, b3BodyId* bodies )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.workerCount = 1;

	// The world registry is global and not thread safe
	b3LockMutex( soak->mutex );
	b3WorldId worldId = b3CreateWorld( &worldDef );
	b3UnlockMutex( soak->mutex );

	CreateGround( worldId, soak->groundMesh );

	b3BoxHull box = b3MakeBoxHull( 0.02f, 0.2f, 0.04f );
	b3Capsule capsule = { { 0.0f, -0.2f, 0.0f }, { 0.0f, 0.2f, 0.0f }, 0.05f };
	b3Sphere sphere = { b3Vec3_zero, 0.05f };

	b3BodyDef bodyDef = b3DefaultBodyDef();
	bodyDef.type = b3_dynamicBody;

	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.baseMaterial.rollingResistance = shapeType == e_capsule || shapeType == e_sphere ? 0.4f : 0.1f;

	// Isolate mesh collision
	shapeDef.filter.categoryBits = 2;
	shapeDef.filter.maskBits = 1;

	uint32_t random = seed;
	int bodyIndex = 0;
	for ( int i = 0; i < GRID_COUNT; ++i )
	{
		for ( int j = 0; j < GRID_COUNT; ++j )
		{
			b3Vec3 linearVelocity = RandomVec3Uniform( &random, -1.0f, 1.0f );
			b3Vec3 angularVelocity = RandomVec3Uniform( &random, -5.0f, 5.0f );

			bodyDef.position = (b3Pos){ 0.5f * ( i - 0.5f * GRID_COUNT ), 5.0f, 0.5f * ( j - 0.5f * GRID_COUNT ) };
			bodyDef.linearVelocity = linearVelocity;
			bodyDef.angularVelocity = angularVelocity;
			b3BodyId bodyId = b3CreateBody( worldId, &bodyDef );

			switch ( shapeType )
			{
				case e_box:
					b3CreateHullShape( bodyId, &shapeDef, &box.base );
					break;
				case e_capsule:
					b3CreateCapsuleShape( bodyId, &shapeDef, &capsule );
					break;
				case e_cylinder:
					b3CreateHullShape( bodyId, &shapeDef, soak->cylinder );
					break;
				default:
					b3CreateSphereShape( bodyId, &shapeDef, &sphere );
					break;
			}

			bodies[bodyIndex++] = bodyId;
		}
	}

	float timeStep = 1.0f / 60.0f;
	int subStepCount = 4;
	int maxSteps = shapeType == e_capsule ? 3000 : 1000;
	int step = 0;
	bool failed = false;

	while ( failed == false )
	{
		b3World_Step( worldId, timeStep, subStepCount );
		step += 1;

		for ( int i = 0; i < BODY_COUNT; ++i )
		{
			b3Pos center = b3Body_GetWorldCenter( bodies[i] );
			if ( center.y < -2.0f )
			{
				Failure failure = { e_tunnel, shapeType, generation, seed, step, i / GRID_COUNT, i % GRID_COUNT, center };
				RecordFailure( soak, &failure );
				failed = true;
				break;
			}
		}

		if ( b3World_GetBodyEvents( worldId ).moveCount == 0 )
		{
			break;
		}

		if ( failed == false && step > maxSteps )
		{
			int awakeIndex = 0;
			for ( int i = 0; i < BODY_COUNT; ++i )
			{
				if ( b3Body_IsAwake( bodies[i] ) )
				{
					awakeIndex = i;
					break;
				}
			}

			b3Pos center = b3Body_GetWorldCenter( bodies[awakeIndex] );
			Failure failure = { e_awake, shapeType, generation, seed, step, awakeIndex / GRID_COUNT, awakeIndex % GRID_COUNT, center };
			RecordFailure( soak, &failure );
			failed = true;
		}
	}

	b3LockMutex( soak->mutex );
	b3DestroyWorld( worldId );
	b3UnlockMutex( soak->mutex );

	return step;
}

static void SoakWorker( void* context )
{
	Soak* soak = context;
	b3BodyId* bodies = malloc( BODY_COUNT * sizeof( b3BodyId ) );

	while ( true )
	{
		b3LockMutex( soak->mutex );
		int generation = soak->nextGeneration;
		soak->nextGeneration += 1;
		b3UnlockMutex( soak->mutex );

		if ( generation >= soak->generationCount )
		{
			break;
		}

		uint32_t seed = soak->fixedSeed ? soak->baseSeed : HashSeed( soak->baseSeed + (uint32_t)generation );
		ShapeType shapeType = soak->shapeType >= 0 ? (ShapeType)soak->shapeType : (ShapeType)( generation % e_shapeCount );

		int steps = RunGeneration( soak, generation, seed, shapeType, bodies );

		b3LockMutex( soak->mutex );
		soak->completedCount += 1;
		soak->totalSteps += steps;
		soak->maxSteps = b3MaxInt( soak->maxSteps, steps );
		int completed = soak->completedCount;
		b3UnlockMutex( soak->mutex );

		if ( completed % 100 == 0 )
		{
			printf( "%d generations, %d bodies\n", completed, completed * BODY_COUNT );
			fflush( stdout );
		}
	}

	free( bodies );
}

static const char* MatchValue( const char* arg, const char* shortName, const char* longName )
{
	size_t shortLength = strlen( shortName );
	if ( strncmp( arg, shortName, shortLength ) == 0 )
	{
		return arg + shortLength;
	}

	size_t longLength = strlen( longName );
	if ( strncmp( arg, longName, longLength ) == 0 )
	{
		return arg + longLength;
	}

	return NULL;
}

int main( int argc, char** argv )
{
	int generationCount = 1000;
	int threadCount = 8;
	int shapeType = e_box;
	uint32_t seed = (uint32_t)b3GetTicks();
	bool fixedSeed = false;
	float amplitude = 0.5f;

	for ( int i = 1; i < argc; ++i )
	{
		const char* arg = argv[i];
		const char* value;

		if ( ( value = MatchValue( arg, "-g=", "--generations=" ) ) != NULL )
		{
			generationCount = atoi( value );
		}
		else if ( ( value = MatchValue( arg, "-t=", "--threads=" ) ) != NULL )
		{
			threadCount = b3ClampInt( atoi( value ), 1, MAX_THREADS );
		}
		else if ( ( value = MatchValue( arg, "-s=", "--shape=" ) ) != NULL )
		{
			shapeType = -2;
			for ( int k = 0; k < e_shapeCount; ++k )
			{
				if ( strcmp( value, s_shapeNames[k] ) == 0 )
				{
					shapeType = k;
				}
			}

			if ( strcmp( value, "all" ) == 0 )
			{
				shapeType = -1;
			}

			if ( shapeType == -2 )
			{
				printf( "unknown shape: %s\n", value );
				return 2;
			}
		}
		else if ( ( value = MatchValue( arg, "-seed=", "--seed=" ) ) != NULL )
		{
			seed = (uint32_t)strtoul( value, NULL, 10 );
			fixedSeed = true;
		}
		else if ( ( value = MatchValue( arg, "-b=", "--base-seed=" ) ) != NULL )
		{
			seed = (uint32_t)strtoul( value, NULL, 10 );
		}
		else if ( ( value = MatchValue( arg, "-a=", "--amplitude=" ) ) != NULL )
		{
			amplitude = (float)atof( value );
		}
		else
		{
			printf( "Usage: mesh_soak [options]\n"
					"Drops %d bodies per generation on the Mesh Drop wave mesh until they sleep.\n"
					"Fails on a body below y = -2 (tunnel) or a pile awake past the step limit.\n"
					"  -g, --generations=<int>  generation count (default 1000)\n"
					"  -t, --threads=<int>      worker threads, one world each (default 8)\n"
					"  -s, --shape=<name>       box, capsule, cylinder, sphere or all (default box)\n"
					"  -b, --base-seed=<uint>   base seed, each generation hashes its own seed from it\n"
					"  --seed=<uint>            run every generation with this exact seed (replay)\n"
					"  -a, --amplitude=<float>  wave amplitude (default 0.5)\n"
					"Exit code is 0 when every generation passes, 1 otherwise.\n",
					BODY_COUNT );
			return 2;
		}
	}

	Soak soak = { 0 };
	soak.mutex = b3CreateMutex();
	soak.groundMesh = b3CreateWaveMesh( 40, 40, 1.0f, amplitude, 0.1f, 0.2f );
	soak.cylinder = b3CreateCylinder( 0.4f, 0.05f, 0.0f, 6 );
	soak.baseSeed = seed;
	soak.fixedSeed = fixedSeed;
	soak.generationCount = generationCount;
	soak.shapeType = shapeType;

	const char* shapeName = shapeType >= 0 ? s_shapeNames[shapeType] : "all";
	printf( "mesh soak: %d generations, %d threads, shape %s, %s seed %u\n", generationCount, threadCount, shapeName,
			fixedSeed ? "fixed" : "base", seed );
	fflush( stdout );

	uint64_t ticks = b3GetTicks();

	b3Thread* threads[MAX_THREADS];
	for ( int i = 0; i < threadCount; ++i )
	{
		threads[i] = b3CreateThread( SoakWorker, &soak, "soak" );
	}

	for ( int i = 0; i < threadCount; ++i )
	{
		b3JoinThread( threads[i] );
	}

	float seconds = 0.001f * b3GetMilliseconds( ticks );
	double averageSteps = soak.completedCount > 0 ? (double)soak.totalSteps / soak.completedCount : 0.0;

	printf( "generations: %d, bodies: %d, tunnels: %d, awake: %d\n", soak.completedCount, soak.completedCount * BODY_COUNT,
			soak.tunnelCount, soak.awakeCount );
	printf( "steps to sleep: average %.0f, max %d, time %.1f s\n", averageSteps, soak.maxSteps, seconds );

	int reported = b3MinInt( soak.failureCount, MAX_REPORTED_FAILURES );
	for ( int i = 0; i < reported; ++i )
	{
		Failure* f = soak.failures + i;
		printf( "  %s %s seed %u body (%d, %d)\n", f->kind == e_tunnel ? "tunnel" : "awake", s_shapeNames[f->shapeType], f->seed,
				f->index1, f->index2 );
	}

	printf( soak.failureCount == 0 ? "PASS\n" : "FAIL\n" );

	b3DestroyHull( soak.cylinder );
	b3DestroyMesh( soak.groundMesh );
	b3DestroyMutex( soak.mutex );

	return soak.failureCount == 0 ? 0 : 1;
}
