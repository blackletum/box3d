// SPDX-FileCopyrightText: 2026 Erin Catto
// SPDX-License-Identifier: MIT

#define b3MeshManifoldWide B3_WIDE( b3MeshManifold )
#define b3MeshConstraintWide B3_WIDE( b3MeshConstraint )

typedef struct b3MeshManifoldWide
{
	int pointCounts[B3_SIMD_WIDTH];

	int pointCount;

	b3Vec3W normal;
	b3Vec3W tangent1;
	b3Vec3W tangent2;

	b3Vec3W centerA, centerB;
	b3FloatW twistMass;
	b3FloatW twistImpulse;
	b3SymMatrix2W tangentMass;
	b3Vec2W frictionImpulse;
	b3Vec3W rollingImpulse;
	b3FloatW rollingMask;
	b3FloatW tangentVelocity1;
	b3FloatW tangentVelocity2;

	b3ContactConstraintPointWide points[B3_MAX_MANIFOLD_POINTS];
} b3MeshManifoldWide;

typedef struct b3MeshConstraintWide
{
	b3FloatW invMassA, invMassB;
	b3SymMatrix3W invIA, invIB;
	b3SymMatrix3W rollingMass;
	b3FloatW friction;
	b3FloatW rollingResistance;
	b3FloatW restitution;
	b3FloatW biasRate, massScale, impulseScale;

	int indexA[B3_SIMD_WIDTH];
	int indexB[B3_SIMD_WIDTH];

	int manifoldCounts[B3_SIMD_WIDTH];
	b3Contact* contacts[B3_SIMD_WIDTH];

	b3MeshManifoldWide* manifolds;
	int manifoldCount;
	int padding[5];
} b3MeshConstraintWide;

int B3_WIDE( b3GetWideMeshConstraintByteCount )( void )
{
	return sizeof( b3MeshConstraintWide );
}

int B3_WIDE( b3GetWideMeshManifoldByteCount )( void )
{
	return sizeof( b3MeshManifoldWide );
}

static const b3Softness b3_zeroSoftness = { 0 };

static inline b3FloatW b3IntsToFloatW( const int* v )
{
#if B3_SIMD_WIDTH == 8
	return b3SetW( (float)v[0], (float)v[1], (float)v[2], (float)v[3], (float)v[4], (float)v[5], (float)v[6], (float)v[7] );
#else
	return b3SetW( (float)v[0], (float)v[1], (float)v[2], (float)v[3] );
#endif
}

void B3_WIDE( b3PrepareContacts_MeshWide )( b3SolverBlock block, b3StepContext* context )
{
	b3TracyCZoneNC( prepare_mesh_wide, "Prepare Mesh", b3_colorYellow, true );

	b3World* world = context->world;
	b3BodySim* sims = context->sims;
	b3BodyState* states = context->states;
#if B3_ENABLE_VALIDATION
	b3Body* bodies = world->bodies.data;
#endif
	b3MeshPrepareSpan* spans = context->meshPrepareSpans;
	b3MeshConstraintWide* wideBase = context->wideMeshConstraints;
	const int* manifoldStarts = context->wideMeshManifoldStarts;

	b3FloatW zeroW = b3ZeroW();
	b3FloatW oneW = b3SplatW( 1.0f );
	b3FloatW twoW = b3SplatW( 2.0f );
	b3FloatW warmStartScale = world->enableWarmStarting ? oneW : zeroW;
	b3FloatW invTau = b3SplatW( 1.0f / B3_SPECULATIVE_DISTANCE );
	b3FloatW minFrictionWeight = b3SplatW( B3_MIN_FRICTION_WEIGHT );
	b3FloatW minDet = b3SplatW( 1000.0f * FLT_MIN );
	bool anyRestitution = false;

	int wideIndex = block.startIndex;
	int endWideIndex = block.startIndex + block.count;

	int colorIndex = 0;
	while ( spans[colorIndex + 1].start <= wideIndex )
	{
		colorIndex += 1;
	}

	while ( wideIndex < endWideIndex )
	{
		int colorWideStart = spans[colorIndex].start;
		int colorWideEndIndex = b3MinInt( spans[colorIndex + 1].start, endWideIndex );
		int colorContactCount = spans[colorIndex].count;
		b3ContactSpec* specs = spans[colorIndex].contacts;
		const int* order = spans[colorIndex].order;

		for ( ; wideIndex < colorWideEndIndex; ++wideIndex )
		{
			b3MeshConstraintWide* c = wideBase + wideIndex;
			int localWideIndex = wideIndex - colorWideStart;

			const b3Contact* contactLanes[B3_SIMD_WIDTH];
			const b3Softness* softLanes[B3_SIMD_WIDTH];
			const b3BodySim* simLanesA[B3_SIMD_WIDTH];
			const b3BodySim* simLanesB[B3_SIMD_WIDTH];
			int hitEventLanes = 0;
			int slotCount = 0;

			for ( int lane = 0; lane < B3_SIMD_WIDTH; ++lane )
			{
				int contactIndex = B3_SIMD_WIDTH * localWideIndex + lane;
				if ( contactIndex < colorContactCount )
				{
					b3ContactSpec* spec = specs + order[contactIndex];
					b3Contact* contact = b3Array_Get( world->contacts, spec->contactId );
					B3_ASSERT( contact->contactId == spec->contactId );
					B3_ASSERT( contact->manifoldCount == spec->manifoldCount );

					int indexA = b3DecodeAwakeIndex( contact->encodedBodySimA );
					int indexB = b3DecodeAwakeIndex( contact->encodedBodySimB );

#if B3_ENABLE_VALIDATION
					b3Body* bodyA = bodies + contact->edges[0].bodyId;
					b3Body* bodyB = bodies + contact->edges[1].bodyId;
					B3_ASSERT( contact->encodedBodySimA == b3EncodeBodySimIndex( bodyA ) );
					B3_ASSERT( contact->encodedBodySimB == b3EncodeBodySimIndex( bodyB ) );
#endif

					c->indexA[lane] = indexA + 1;
					c->indexB[lane] = indexB + 1;
					c->manifoldCounts[lane] = contact->manifoldCount;
					c->contacts[lane] = contact;

					contactLanes[lane] = contact;
					softLanes[lane] =
						( contact->flags & b3_contactStaticFlag ) != 0 ? &context->staticSoftness : &context->contactSoftness;
					simLanesA[lane] = indexA == B3_NULL_INDEX ? &b3_zeroBodySim : sims + indexA;
					simLanesB[lane] = indexB == B3_NULL_INDEX ? &b3_zeroBodySim : sims + indexB;
					hitEventLanes |= ( contact->flags & b3_simEnableHitEvent ) != 0 ? 1 << lane : 0;
					slotCount = b3MaxInt( slotCount, contact->manifoldCount );
				}
				else
				{
					c->indexA[lane] = 0;
					c->indexB[lane] = 0;
					c->manifoldCounts[lane] = 0;
					c->contacts[lane] = NULL;

					contactLanes[lane] = &b3_zeroContact;
					softLanes[lane] = &b3_zeroSoftness;
					simLanesA[lane] = &b3_zeroBodySim;
					simLanesB[lane] = &b3_zeroBodySim;
				}
			}

			int slotStart = manifoldStarts[wideIndex];
			c->manifoldCount = manifoldStarts[wideIndex + 1] - slotStart;
			c->manifolds = (b3MeshManifoldWide*)context->wideMeshManifolds + slotStart;
			B3_VALIDATE( c->manifoldCount == slotCount );
			B3_UNUSED( slotCount );

			b3FloatW mA, mB;
			B3_GATHER_LANES( mA, simLanesA, invMass );
			B3_GATHER_LANES( mB, simLanesB, invMass );
			b3SymMatrix3W iA = b3GatherInvInertiaW( simLanesA );
			b3SymMatrix3W iB = b3GatherInvInertiaW( simLanesB );
			c->invMassA = mA;
			c->invMassB = mB;
			c->invIA = iA;
			c->invIB = iB;

			b3Vec3W tangentVelocity;
			B3_GATHER_LANES( c->friction, contactLanes, friction );
			B3_GATHER_LANES( c->rollingResistance, contactLanes, rollingResistance );
			B3_GATHER_LANES( c->restitution, contactLanes, restitution );
			B3_GATHER_LANES( tangentVelocity.X, contactLanes, tangentVelocity.x );
			B3_GATHER_LANES( tangentVelocity.Y, contactLanes, tangentVelocity.y );
			B3_GATHER_LANES( tangentVelocity.Z, contactLanes, tangentVelocity.z );
			B3_GATHER_LANES( c->massScale, softLanes, massScale );
			B3_GATHER_LANES( c->impulseScale, softLanes, impulseScale );
			B3_GATHER_LANES( c->biasRate, softLanes, biasRate );
			c->biasRate = b3MulW( c->massScale, c->biasRate );

			b3SymMatrix3W invIAB = b3AddSymW( iA, iB );
			bool anyRolling = b3AllZeroW( c->rollingResistance ) == false;
			if ( anyRolling )
			{
				c->rollingMass = b3InvertSymW( invIAB );
			}
			else
			{
				c->rollingMass = (b3SymMatrix3W){ zeroW, zeroW, zeroW, zeroW, zeroW, zeroW };
			}

			b3FloatW rollingResistanceMask = b3GreaterThanW( c->rollingResistance, zeroW );

			bool haveRestitution = b3AnyTrueW( b3GreaterThanW( c->restitution, zeroW ) );
			anyRestitution = anyRestitution || haveRestitution;
			bool sampleVelocity = hitEventLanes != 0 || haveRestitution;

			for ( int slotIndex = 0; slotIndex < c->manifoldCount; ++slotIndex )
			{
				b3MeshManifoldWide* slot = c->manifolds + slotIndex;

				const b3Manifold* manifoldLanes[B3_SIMD_WIDTH];
				int activeLanes[B3_SIMD_WIDTH];
				int pointCount = 0;
				for ( int lane = 0; lane < B3_SIMD_WIDTH; ++lane )
				{
					if ( slotIndex < c->manifoldCounts[lane] )
					{
						const b3Manifold* manifold = contactLanes[lane]->manifolds + slotIndex;
						manifoldLanes[lane] = manifold;
						activeLanes[lane] = 1;
						slot->pointCounts[lane] = manifold->pointCount;
					}
					else
					{
						manifoldLanes[lane] = &b3_zeroManifold;
						activeLanes[lane] = 0;
						slot->pointCounts[lane] = 0;
					}

					pointCount = b3MaxInt( pointCount, slot->pointCounts[lane] );
				}

				B3_VALIDATE( 0 <= pointCount && pointCount <= B3_MAX_MANIFOLD_POINTS );
				slot->pointCount = pointCount;

				slot->rollingMask = b3AndW( rollingResistanceMask, b3GreaterThanW( b3IntsToFloatW( activeLanes ), zeroW ) );

				b3Vec3W normal, frictionImpulse, rollingImpulse;
				b3FloatW twistImpulse;
				{
					const float* m0 = &manifoldLanes[0]->normal.x;
					const float* m1 = &manifoldLanes[1]->normal.x;
					const float* m2 = &manifoldLanes[2]->normal.x;
					const float* m3 = &manifoldLanes[3]->normal.x;
#if B3_SIMD_WIDTH == 8
					const float* m4 = &manifoldLanes[4]->normal.x;
					const float* m5 = &manifoldLanes[5]->normal.x;
					const float* m6 = &manifoldLanes[6]->normal.x;
					const float* m7 = &manifoldLanes[7]->normal.x;
					b3TransposeW( b3LoadW( m0 ), b3LoadW( m1 ), b3LoadW( m2 ), b3LoadW( m3 ), b3LoadW( m4 ), b3LoadW( m5 ),
								  b3LoadW( m6 ), b3LoadW( m7 ), &normal.X, &normal.Y, &normal.Z, &twistImpulse, &frictionImpulse.X,
								  &frictionImpulse.Y, &frictionImpulse.Z, &rollingImpulse.X );
					rollingImpulse.Y = b3SetW( m0[8], m1[8], m2[8], m3[8], m4[8], m5[8], m6[8], m7[8] );
					rollingImpulse.Z = b3SetW( m0[9], m1[9], m2[9], m3[9], m4[9], m5[9], m6[9], m7[9] );
#else
					b3TransposeW( b3LoadW( m0 ), b3LoadW( m1 ), b3LoadW( m2 ), b3LoadW( m3 ), &normal.X, &normal.Y, &normal.Z,
								  &twistImpulse );
					b3TransposeW( b3LoadW( m0 + 4 ), b3LoadW( m1 + 4 ), b3LoadW( m2 + 4 ), b3LoadW( m3 + 4 ), &frictionImpulse.X,
								  &frictionImpulse.Y, &frictionImpulse.Z, &rollingImpulse.X );
					rollingImpulse.Y = b3SetW( m0[8], m1[8], m2[8], m3[8] );
					rollingImpulse.Z = b3SetW( m0[9], m1[9], m2[9], m3[9] );
#endif
				}

				b3Vec3W tangent1 = b3PerpW( normal );
				b3Vec3W tangent2 = b3CrossW( tangent1, normal );
				slot->normal = normal;
				slot->tangent1 = tangent1;
				slot->tangent2 = tangent2;
				slot->tangentVelocity1 = b3DotW( tangentVelocity, tangent1 );
				slot->tangentVelocity2 = b3DotW( tangentVelocity, tangent2 );

				slot->twistImpulse = b3MulW( warmStartScale, twistImpulse );
				slot->rollingImpulse.X = b3BlendW( zeroW, b3MulW( warmStartScale, rollingImpulse.X ), slot->rollingMask );
				slot->rollingImpulse.Y = b3BlendW( zeroW, b3MulW( warmStartScale, rollingImpulse.Y ), slot->rollingMask );
				slot->rollingImpulse.Z = b3BlendW( zeroW, b3MulW( warmStartScale, rollingImpulse.Z ), slot->rollingMask );
				slot->frictionImpulse.x = b3MulW( warmStartScale, b3DotW( frictionImpulse, tangent1 ) );
				slot->frictionImpulse.y = b3MulW( warmStartScale, b3DotW( frictionImpulse, tangent2 ) );

				b3FloatW pointCountW = b3IntsToFloatW( slot->pointCounts );

				b3Vec3W centerA = { zeroW, zeroW, zeroW };
				b3Vec3W centerB = { zeroW, zeroW, zeroW };
				b3FloatW totalFrictionWeight = zeroW;

				for ( int pointIndex = 0; pointIndex < pointCount; ++pointIndex )
				{
					b3ContactConstraintPointWide* cp = slot->points + pointIndex;
					b3FloatW pointMask = b3GreaterThanW( pointCountW, b3SplatW( (float)pointIndex ) );

					const float* p0 = (const float*)( manifoldLanes[0]->points + pointIndex );
					const float* p1 = (const float*)( manifoldLanes[1]->points + pointIndex );
					const float* p2 = (const float*)( manifoldLanes[2]->points + pointIndex );
					const float* p3 = (const float*)( manifoldLanes[3]->points + pointIndex );
#if B3_SIMD_WIDTH == 8
					const float* p4 = (const float*)( manifoldLanes[4]->points + pointIndex );
					const float* p5 = (const float*)( manifoldLanes[5]->points + pointIndex );
					const float* p6 = (const float*)( manifoldLanes[6]->points + pointIndex );
					const float* p7 = (const float*)( manifoldLanes[7]->points + pointIndex );
#endif

					b3Vec3W rA, rB;
					b3FloatW separation, normalImpulse;
#if B3_SIMD_WIDTH == 8
					b3TransposeW( b3LoadW( p0 ), b3LoadW( p1 ), b3LoadW( p2 ), b3LoadW( p3 ), b3LoadW( p4 ), b3LoadW( p5 ),
								  b3LoadW( p6 ), b3LoadW( p7 ), &rA.X, &rA.Y, &rA.Z, &rB.X, &rB.Y, &rB.Z, &separation,
								  &normalImpulse );
#else
					b3TransposeW( b3LoadW( p0 ), b3LoadW( p1 ), b3LoadW( p2 ), b3LoadW( p3 ), &rA.X, &rA.Y, &rA.Z, &rB.X );
					b3TransposeW( b3LoadW( p0 + 4 ), b3LoadW( p1 + 4 ), b3LoadW( p2 + 4 ), b3LoadW( p3 + 4 ), &rB.Y, &rB.Z,
								  &separation, &normalImpulse );
#endif

					rA.X = b3BlendW( zeroW, rA.X, pointMask );
					rA.Y = b3BlendW( zeroW, rA.Y, pointMask );
					rA.Z = b3BlendW( zeroW, rA.Z, pointMask );
					rB.X = b3BlendW( zeroW, rB.X, pointMask );
					rB.Y = b3BlendW( zeroW, rB.Y, pointMask );
					rB.Z = b3BlendW( zeroW, rB.Z, pointMask );
					separation = b3BlendW( zeroW, separation, pointMask );
					normalImpulse = b3BlendW( zeroW, normalImpulse, pointMask );

					b3FloatW weight = b3MinW( b3MaxW( b3SubW( twoW, b3MulW( separation, invTau ) ), minFrictionWeight ), oneW );
					weight = b3BlendW( zeroW, weight, pointMask );
					centerA = b3MulAddSVW( centerA, weight, rA );
					centerB = b3MulAddSVW( centerB, weight, rB );
					totalFrictionWeight = b3AddW( totalFrictionWeight, weight );

					cp->anchorAs = rA;
					cp->anchorBs = rB;
					cp->baseSeparations = b3SubW( separation, b3DotW( b3SubVW( rB, rA ), normal ) );
					cp->normalImpulses = b3MulW( warmStartScale, normalImpulse );
					cp->totalNormalImpulses = zeroW;
					cp->relativeVelocities = zeroW;
					cp->restitutionImpulses = zeroW;

					b3Vec3W rnA = b3CrossW( rA, normal );
					b3Vec3W rnB = b3CrossW( rB, normal );
					b3FloatW kNormal = b3AddW( mA, mB );
					kNormal = b3AddW( kNormal, b3DotW( rnA, b3MulMVW( iA, rnA ) ) );
					kNormal = b3AddW( kNormal, b3DotW( rnB, b3MulMVW( iB, rnB ) ) );
					b3FloatW valid = b3AndW( b3GreaterThanW( kNormal, zeroW ), pointMask );
					cp->normalMasses = b3BlendW( zeroW, b3DivW( oneW, kNormal ), valid );
				}

				b3FloatW frictionValid = b3GreaterThanW( totalFrictionWeight, zeroW );
				b3FloatW invWeight = b3BlendW( zeroW, b3DivW( oneW, totalFrictionWeight ), frictionValid );
				centerA = b3MulSVW( invWeight, centerA );
				centerB = b3MulSVW( invWeight, centerB );
				slot->centerA = centerA;
				slot->centerB = centerB;

				for ( int pointIndex = 0; pointIndex < pointCount; ++pointIndex )
				{
					b3ContactConstraintPointWide* cp = slot->points + pointIndex;
					b3FloatW pointMask = b3GreaterThanW( pointCountW, b3SplatW( (float)pointIndex ) );
					b3Vec3W d = b3SubVW( cp->anchorAs, centerA );
					cp->leverArms = b3BlendW( zeroW, b3SqrtW( b3DotW( d, d ) ), pointMask );
				}

				{
					b3Vec3W rtA1 = b3CrossW( centerA, tangent1 );
					b3Vec3W rtA2 = b3CrossW( centerA, tangent2 );
					b3Vec3W rtB1 = b3CrossW( centerB, tangent1 );
					b3Vec3W rtB2 = b3CrossW( centerB, tangent2 );
					b3Vec3W iArtA1 = b3MulMVW( iA, rtA1 );
					b3Vec3W iArtA2 = b3MulMVW( iA, rtA2 );
					b3Vec3W iBrtB1 = b3MulMVW( iB, rtB1 );
					b3Vec3W iBrtB2 = b3MulMVW( iB, rtB2 );

					b3FloatW kxx = b3AddW( b3AddW( b3AddW( mA, mB ), b3DotW( rtA1, iArtA1 ) ), b3DotW( rtB1, iBrtB1 ) );
					b3FloatW kyy = b3AddW( b3AddW( b3AddW( mA, mB ), b3DotW( rtA2, iArtA2 ) ), b3DotW( rtB2, iBrtB2 ) );
					b3FloatW kxy = b3AddW( b3DotW( rtA1, iArtA2 ), b3DotW( rtB1, iBrtB2 ) );

					b3FloatW det = b3SubW( b3MulW( kxx, kyy ), b3MulW( kxy, kxy ) );
					b3FloatW valid = b3AndW( b3GreaterThanW( b3AbsW( det ), minDet ), frictionValid );
					b3FloatW invDet = b3BlendW( zeroW, b3DivW( oneW, det ), valid );
					slot->tangentMass.cxx = b3MulW( invDet, kyy );
					slot->tangentMass.cxy = b3NegW( b3MulW( invDet, kxy ) );
					slot->tangentMass.cyy = b3MulW( invDet, kxx );
				}

				{
					b3FloatW kTwist = b3DotW( normal, b3MulMVW( invIAB, normal ) );
					slot->twistMass = b3BlendW( zeroW, b3DivW( oneW, kTwist ), b3GreaterThanW( kTwist, zeroW ) );
				}
			}

			if ( sampleVelocity )
			{
				b3BodyStateW bA = b3GatherBodies( states, c->indexA );
				b3BodyStateW bB = b3GatherBodies( states, c->indexB );

				for ( int slotIndex = 0; slotIndex < c->manifoldCount; ++slotIndex )
				{
					b3MeshManifoldWide* slot = c->manifolds + slotIndex;

					for ( int pointIndex = 0; pointIndex < slot->pointCount; ++pointIndex )
					{
						b3ContactConstraintPointWide* cp = slot->points + pointIndex;

						b3Vec3W vrA = b3AddVW( bA.v, b3CrossW( bA.w, cp->anchorAs ) );
						b3Vec3W vrB = b3AddVW( bB.v, b3CrossW( bB.w, cp->anchorBs ) );
						b3FloatW vn = b3DotW( slot->normal, b3SubVW( vrB, vrA ) );
						cp->relativeVelocities = vn;

						if ( hitEventLanes != 0 )
						{
							float normalVelocities[B3_SIMD_WIDTH];
							b3StoreW( normalVelocities, vn );

							for ( int lane = 0; lane < B3_SIMD_WIDTH; ++lane )
							{
								if ( ( hitEventLanes & ( 1 << lane ) ) != 0 && pointIndex < slot->pointCounts[lane] )
								{
									c->contacts[lane]->manifolds[slotIndex].points[pointIndex].normalVelocity =
										normalVelocities[lane];
								}
							}
						}
					}
				}
			}
		}

		colorIndex += 1;
	}

	if ( anyRestitution )
	{
		b3AtomicStoreInt( &context->anyRestitution, 1 );
	}

	b3TracyCZoneEnd( prepare_mesh_wide );
}

void B3_WIDE( b3WarmStartContacts_MeshWide )( b3SolverBlock block, b3StepContext* context )
{
	b3TracyCZoneNC( warm_start_mesh_wide, "Warm Start Mesh", b3_colorGreen, true );

	b3BodyState* states = context->states;
	b3MeshConstraintWide* constraints = context->graph->colors[block.colorIndex].wideMeshConstraints;

	b3FloatW zeroW = b3ZeroW();

	for ( int i = block.startIndex; i < block.startIndex + block.count; ++i )
	{
		b3MeshConstraintWide* c = constraints + i;
		b3BodyStateW bA = b3GatherBodies( states, c->indexA );
		b3BodyStateW bB = b3GatherBodies( states, c->indexB );

		bool anyRolling = b3AllZeroW( c->rollingResistance ) == false;

		b3Vec3W linearImpulse = { zeroW, zeroW, zeroW };
		b3Vec3W angularImpulseA = { zeroW, zeroW, zeroW };
		b3Vec3W angularImpulseB = { zeroW, zeroW, zeroW };

		for ( int slotIndex = 0; slotIndex < c->manifoldCount; ++slotIndex )
		{
			const b3MeshManifoldWide* slot = c->manifolds + slotIndex;

			b3FloatW totalNormalImpulse = zeroW;
			b3Vec3W momentA = { zeroW, zeroW, zeroW };
			b3Vec3W momentB = { zeroW, zeroW, zeroW };

			for ( int pointIndex = 0; pointIndex < slot->pointCount; ++pointIndex )
			{
				const b3ContactConstraintPointWide* cp = slot->points + pointIndex;
				b3FloatW normalImpulse = cp->normalImpulses;
				totalNormalImpulse = b3AddW( totalNormalImpulse, normalImpulse );
				momentA = b3MulAddSVW( momentA, normalImpulse, cp->anchorAs );
				momentB = b3MulAddSVW( momentB, normalImpulse, cp->anchorBs );
			}

			b3Vec3W normal = slot->normal;
			b3Vec3W frictionImpulse = b3MulSVW( slot->frictionImpulse.x, slot->tangent1 );
			frictionImpulse = b3MulAddSVW( frictionImpulse, slot->frictionImpulse.y, slot->tangent2 );

			linearImpulse = b3AddVW( linearImpulse, b3MulAddSVW( frictionImpulse, totalNormalImpulse, normal ) );

			b3Vec3W twistImpulse = b3MulSVW( slot->twistImpulse, normal );
			angularImpulseA = b3AddVW(
				angularImpulseA,
				b3AddVW( b3AddVW( b3CrossW( momentA, normal ), b3CrossW( slot->centerA, frictionImpulse ) ), twistImpulse ) );
			angularImpulseB = b3AddVW(
				angularImpulseB,
				b3AddVW( b3AddVW( b3CrossW( momentB, normal ), b3CrossW( slot->centerB, frictionImpulse ) ), twistImpulse ) );

			if ( anyRolling )
			{
				angularImpulseA = b3AddVW( angularImpulseA, slot->rollingImpulse );
				angularImpulseB = b3AddVW( angularImpulseB, slot->rollingImpulse );
			}
		}

		bA.w = b3MulSubMVW( bA.w, c->invIA, angularImpulseA );
		bA.v = b3MulSubSVW( bA.v, c->invMassA, linearImpulse );
		bB.w = b3MulAddMVW( bB.w, c->invIB, angularImpulseB );
		bB.v = b3MulAddSVW( bB.v, c->invMassB, linearImpulse );

		b3ScatterBodies( states, c->indexA, &bA );
		b3ScatterBodies( states, c->indexB, &bB );
	}

	b3TracyCZoneEnd( warm_start_mesh_wide );
}

void B3_WIDE( b3PushContacts_MeshWide )( b3SolverBlock block, b3StepContext* context )
{
	b3TracyCZoneNC( push_mesh_wide, "Push Mesh", b3_colorAliceBlue, true );

	b3BodyState* states = context->states;
	b3MeshConstraintWide* constraints = context->graph->colors[block.colorIndex].wideMeshConstraints;
	b3FloatW inv_h = b3SplatW( context->inv_h );
	b3FloatW contactSpeed = b3SplatW( -context->world->contactSpeed );
	b3FloatW oneW = b3SplatW( 1.0f );
	b3FloatW zeroW = b3ZeroW();

	for ( int wideIndex = block.startIndex; wideIndex < block.startIndex + block.count; ++wideIndex )
	{
		b3MeshConstraintWide* c = constraints + wideIndex;

		b3BodyStateW bA = b3GatherBodies( states, c->indexA );
		b3BodyStateW bB = b3GatherBodies( states, c->indexB );

		b3FloatW biasRate = c->biasRate;
		b3FloatW massScale = c->massScale;
		b3FloatW impulseScale = c->impulseScale;

		b3Vec3W dp = b3SubVW( bB.dp, bA.dp );

		for ( int slotIndex = 0; slotIndex < c->manifoldCount; ++slotIndex )
		{
			b3MeshManifoldWide* slot = c->manifolds + slotIndex;

			b3Vec3W normal = slot->normal;
			b3FloatW normalSeparation = b3DotW( normal, dp );
			b3Vec3W normalA = b3InvRotateVectorW( bA.dq, normal );
			b3Vec3W normalB = b3InvRotateVectorW( bB.dq, normal );

			for ( int pointIndex = 0; pointIndex < slot->pointCount; ++pointIndex )
			{
				b3ContactConstraintPointWide* cp = slot->points + pointIndex;

				b3Vec3W rA = cp->anchorAs;
				b3Vec3W rB = cp->anchorBs;

				b3FloatW s =
					b3AddW( b3AddW( normalSeparation, b3SubW( b3DotW( normalB, rB ), b3DotW( normalA, rA ) ) ), cp->baseSeparations );

				b3FloatW separated = b3GreaterThanW( s, zeroW );

				b3FloatW specBias = b3MulW( s, inv_h );
				b3FloatW overlapBias = b3MaxW( b3MulW( biasRate, s ), contactSpeed );
				b3FloatW velocityBias = b3BlendW( overlapBias, specBias, separated );

				b3FloatW pointMassScale = b3BlendW( massScale, oneW, separated );
				b3FloatW pointImpulseScale = b3BlendW( impulseScale, zeroW, separated );

				b3Vec3W vrA = b3AddVW( bA.v, b3CrossW( bA.w, rA ) );
				b3Vec3W vrB = b3AddVW( bB.v, b3CrossW( bB.w, rB ) );
				b3FloatW vn = b3DotW( b3SubVW( vrB, vrA ), normal );

				b3FloatW negImpulse = b3AddW( b3MulW( cp->normalMasses, b3AddW( b3MulW( pointMassScale, vn ), velocityBias ) ),
											  b3MulW( pointImpulseScale, cp->normalImpulses ) );

				b3FloatW newImpulse = b3MaxW( b3SubW( cp->normalImpulses, negImpulse ), zeroW );
				b3FloatW deltaImpulse = b3SubW( newImpulse, cp->normalImpulses );
				cp->normalImpulses = newImpulse;

				b3Vec3W P = b3MulSVW( deltaImpulse, normal );
				bA.w = b3MulSubMVW( bA.w, c->invIA, b3CrossW( rA, P ) );
				bA.v = b3MulSubSVW( bA.v, c->invMassA, P );
				bB.w = b3MulAddMVW( bB.w, c->invIB, b3CrossW( rB, P ) );
				bB.v = b3MulAddSVW( bB.v, c->invMassB, P );
			}
		}

		b3ScatterBodies( states, c->indexA, &bA );
		b3ScatterBodies( states, c->indexB, &bB );
	}

	b3TracyCZoneEnd( push_mesh_wide );
}

void B3_WIDE( b3SolveContacts_MeshWide )( b3SolverBlock block, b3StepContext* context )
{
	b3TracyCZoneNC( solve_mesh_wide, "Solve Mesh", b3_colorAliceBlue, true );

	b3BodyState* states = context->states;
	b3MeshConstraintWide* constraints = context->graph->colors[block.colorIndex].wideMeshConstraints;
	b3FloatW inv_h = b3SplatW( context->inv_h );
	b3FloatW oneW = b3SplatW( 1.0f );
	b3FloatW zeroW = b3ZeroW();
	b3FloatW epsilonW = b3SplatW( FLT_EPSILON );

	for ( int wideIndex = block.startIndex; wideIndex < block.startIndex + block.count; ++wideIndex )
	{
		b3MeshConstraintWide* c = constraints + wideIndex;

		b3BodyStateW bA = b3GatherBodies( states, c->indexA );
		b3BodyStateW bB = b3GatherBodies( states, c->indexB );

		bool anyRolling = b3AllZeroW( c->rollingResistance ) == false;
		b3Vec3W dp = b3SubVW( bB.dp, bA.dp );

		for ( int slotIndex = 0; slotIndex < c->manifoldCount; ++slotIndex )
		{
			b3MeshManifoldWide* slot = c->manifolds + slotIndex;

			b3Vec3W normal = slot->normal;
			b3FloatW normalSeparation = b3DotW( normal, dp );
			b3Vec3W normalA = b3InvRotateVectorW( bA.dq, normal );
			b3Vec3W normalB = b3InvRotateVectorW( bB.dq, normal );

			b3FloatW totalNormalImpulse = zeroW;
			b3FloatW totalTwistLimit = zeroW;

			for ( int pointIndex = 0; pointIndex < slot->pointCount; ++pointIndex )
			{
				b3ContactConstraintPointWide* cp = slot->points + pointIndex;

				b3Vec3W rA = cp->anchorAs;
				b3Vec3W rB = cp->anchorBs;

				b3FloatW s =
					b3AddW( b3AddW( normalSeparation, b3SubW( b3DotW( normalB, rB ), b3DotW( normalA, rA ) ) ), cp->baseSeparations );

				b3FloatW velocityBias = b3MaxW( zeroW, b3MulW( s, inv_h ) );

				b3Vec3W vrA = b3AddVW( bA.v, b3CrossW( bA.w, rA ) );
				b3Vec3W vrB = b3AddVW( bB.v, b3CrossW( bB.w, rB ) );
				b3FloatW vn = b3DotW( b3SubVW( vrB, vrA ), normal );

				b3FloatW negImpulse = b3MulW( cp->normalMasses, b3AddW( vn, velocityBias ) );

				b3FloatW newImpulse = b3MaxW( b3SubW( cp->normalImpulses, negImpulse ), zeroW );
				b3FloatW deltaImpulse = b3SubW( newImpulse, cp->normalImpulses );
				cp->normalImpulses = newImpulse;
				cp->totalNormalImpulses = b3AddW( cp->totalNormalImpulses, newImpulse );
				totalNormalImpulse = b3AddW( totalNormalImpulse, newImpulse );
				totalTwistLimit = b3AddW( totalTwistLimit, b3MulW( cp->leverArms, newImpulse ) );

				b3Vec3W P = b3MulSVW( deltaImpulse, normal );
				bA.w = b3MulSubMVW( bA.w, c->invIA, b3CrossW( rA, P ) );
				bA.v = b3MulSubSVW( bA.v, c->invMassA, P );
				bB.w = b3MulAddMVW( bB.w, c->invIB, b3CrossW( rB, P ) );
				bB.v = b3MulAddSVW( bB.v, c->invMassB, P );
			}

			{
				b3FloatW twistSpeed = b3DotW( normal, b3SubVW( bB.w, bA.w ) );
				b3FloatW maxLambda = b3MulW( c->friction, totalTwistLimit );
				b3FloatW deltaImpulse = b3NegW( b3MulW( slot->twistMass, twistSpeed ) );
				b3FloatW oldImpulse = slot->twistImpulse;
				slot->twistImpulse = b3SymClampW( b3AddW( oldImpulse, deltaImpulse ), maxLambda );
				deltaImpulse = b3SubW( slot->twistImpulse, oldImpulse );

				b3Vec3W L = b3MulSVW( deltaImpulse, normal );
				bA.w = b3MulSubMVW( bA.w, c->invIA, L );
				bB.w = b3MulAddMVW( bB.w, c->invIB, L );
			}

			if ( anyRolling )
			{
				b3Vec3W deltaImpulse = b3MulMVW( c->rollingMass, b3SubVW( bA.w, bB.w ) );
				b3Vec3W oldImpulse = slot->rollingImpulse;
				slot->rollingImpulse = b3AddVW( oldImpulse, deltaImpulse );

				b3FloatW maxImpulse = b3MulW( c->rollingResistance, totalNormalImpulse );
				b3FloatW lengthSquared = b3DotW( slot->rollingImpulse, slot->rollingImpulse );

				b3FloatW mask = b3GreaterThanW( lengthSquared, b3MulAddW( epsilonW, maxImpulse, maxImpulse ) );

				b3FloatW normalize = b3DivW( maxImpulse, b3AddW( b3SqrtW( lengthSquared ), epsilonW ) );
				b3FloatW scale = b3BlendW( oneW, normalize, mask );

				b3Vec3W scaledImpulse = b3MulSVW( scale, slot->rollingImpulse );
				slot->rollingImpulse.X = b3BlendW( zeroW, scaledImpulse.X, slot->rollingMask );
				slot->rollingImpulse.Y = b3BlendW( zeroW, scaledImpulse.Y, slot->rollingMask );
				slot->rollingImpulse.Z = b3BlendW( zeroW, scaledImpulse.Z, slot->rollingMask );

				deltaImpulse = b3SubVW( slot->rollingImpulse, oldImpulse );

				bA.w = b3MulSubMVW( bA.w, c->invIA, deltaImpulse );
				bB.w = b3MulAddMVW( bB.w, c->invIB, deltaImpulse );
			}

			{
				b3Vec3W tangent1 = slot->tangent1;
				b3Vec3W tangent2 = slot->tangent2;

				b3Vec3W rA = slot->centerA;
				b3Vec3W rB = slot->centerB;

				b3Vec3W vrA = b3AddVW( bA.v, b3CrossW( bA.w, rA ) );
				b3Vec3W vrB = b3AddVW( bB.v, b3CrossW( bB.w, rB ) );
				b3Vec3W vr = b3SubVW( vrB, vrA );
				b3Vec2W vt = {
					b3SubW( b3DotW( vr, tangent1 ), slot->tangentVelocity1 ),
					b3SubW( b3DotW( vr, tangent2 ), slot->tangentVelocity2 ),
				};

				b3Vec2W deltaImpulse = b3MulMV2W( slot->tangentMass, vt );
				deltaImpulse = (b3Vec2W){ b3NegW( deltaImpulse.x ), b3NegW( deltaImpulse.y ) };
				b3Vec2W newImpulse = b3AddV2W( slot->frictionImpulse, deltaImpulse );

				b3FloatW maxImpulse = b3MulW( c->friction, totalNormalImpulse );

				b3FloatW lengthSquared = b3AddW( b3MulW( newImpulse.x, newImpulse.x ), b3MulW( newImpulse.y, newImpulse.y ) );

				b3FloatW mask = b3GreaterThanW( lengthSquared, b3MulW( maxImpulse, maxImpulse ) );

				b3FloatW normalize = b3DivW( maxImpulse, b3AddW( b3SqrtW( lengthSquared ), epsilonW ) );
				b3FloatW scale = b3BlendW( oneW, normalize, mask );
				newImpulse = (b3Vec2W){
					b3MulW( scale, newImpulse.x ),
					b3MulW( scale, newImpulse.y ),
				};

				deltaImpulse = (b3Vec2W){
					b3SubW( newImpulse.x, slot->frictionImpulse.x ),
					b3SubW( newImpulse.y, slot->frictionImpulse.y ),
				};

				slot->frictionImpulse = newImpulse;

				b3Vec3W P = b3AddVW( b3MulSVW( deltaImpulse.x, tangent1 ), b3MulSVW( deltaImpulse.y, tangent2 ) );
				bA.w = b3MulSubMVW( bA.w, c->invIA, b3CrossW( rA, P ) );
				bA.v = b3MulSubSVW( bA.v, c->invMassA, P );
				bB.w = b3MulAddMVW( bB.w, c->invIB, b3CrossW( rB, P ) );
				bB.v = b3MulAddSVW( bB.v, c->invMassB, P );
			}
		}

		b3ScatterBodies( states, c->indexA, &bA );
		b3ScatterBodies( states, c->indexB, &bB );
	}

	b3TracyCZoneEnd( solve_mesh_wide );
}

void B3_WIDE( b3ApplyRestitution_MeshWide )( b3SolverBlock block, b3StepContext* context )
{
	b3TracyCZoneNC( restitution_mesh_wide, "Restitution Mesh", b3_colorDodgerBlue, true );

	b3BodyState* states = context->states;
	b3MeshConstraintWide* constraints = context->graph->colors[block.colorIndex].wideMeshConstraints;
	b3FloatW inv_h = b3SplatW( context->inv_h );
	b3FloatW negRestitutionThreshold = b3SplatW( -context->world->restitutionThreshold );
	b3FloatW zeroW = b3ZeroW();
	bool propagate = context->world->enableRestitutionPropagation;

	for ( int wideIndex = block.startIndex; wideIndex < block.startIndex + block.count; ++wideIndex )
	{
		b3MeshConstraintWide* c = constraints + wideIndex;
		if ( propagate == false && b3AllZeroW( c->restitution ) )
		{
			continue;
		}

		b3BodyStateW bA = b3GatherBodies( states, c->indexA );
		b3BodyStateW bB = b3GatherBodies( states, c->indexB );

		b3FloatW restitutionMask = b3GreaterThanW( c->restitution, zeroW );
		b3Vec3W dp = b3SubVW( bB.dp, bA.dp );
		b3Matrix3W dqA = b3MakeMatrixFromQuatW( bA.dq );
		b3Matrix3W dqB = b3MakeMatrixFromQuatW( bB.dq );

		for ( int slotIndex = 0; slotIndex < c->manifoldCount; ++slotIndex )
		{
			b3MeshManifoldWide* slot = c->manifolds + slotIndex;
			b3Vec3W normal = slot->normal;

			for ( int pointIndex = 0; pointIndex < slot->pointCount; ++pointIndex )
			{
				b3ContactConstraintPointWide* cp = slot->points + pointIndex;

				b3Vec3W rA = cp->anchorAs;
				b3Vec3W rB = cp->anchorBs;

				b3FloatW normalMass = propagate ? cp->normalMasses : b3BlendW( zeroW, cp->normalMasses, restitutionMask );

				b3FloatW compressionImpulse = b3SubW( cp->totalNormalImpulses, cp->restitutionImpulses );
				b3FloatW armed = b3AndW( b3AndW( restitutionMask, b3LessThanW( cp->relativeVelocities, negRestitutionThreshold ) ),
										 b3GreaterThanW( compressionImpulse, zeroW ) );

				b3Vec3W rsA = b3MulM3VW( dqA, rA );
				b3Vec3W rsB = b3MulM3VW( dqB, rB );
				b3Vec3W ds = b3AddVW( dp, b3SubVW( rsB, rsA ) );
				b3FloatW s = b3AddW( b3DotW( normal, ds ), cp->baseSeparations );

				b3FloatW specBias = b3MaxW( zeroW, b3MulW( s, inv_h ) );
				b3FloatW velocityBias = b3BlendW( specBias, b3MulW( c->restitution, cp->relativeVelocities ), armed );

				b3Vec3W vrA = b3AddVW( bA.v, b3CrossW( bA.w, rA ) );
				b3Vec3W vrB = b3AddVW( bB.v, b3CrossW( bB.w, rB ) );
				b3FloatW vn = b3DotW( b3SubVW( vrB, vrA ), normal );

				b3FloatW negImpulse = b3MulW( normalMass, b3AddW( vn, velocityBias ) );

				b3FloatW newImpulse = b3MaxW( b3SubW( cp->normalImpulses, negImpulse ), zeroW );
				b3FloatW impulse = b3SubW( newImpulse, cp->normalImpulses );

				b3FloatW approachImpulse = b3MinW( b3MaxW( b3NegW( b3MulW( normalMass, vn ) ), zeroW ), b3MaxW( impulse, zeroW ) );
				b3FloatW allowance =
					b3SubW( b3MulW( c->restitution, b3AddW( compressionImpulse, approachImpulse ) ), cp->restitutionImpulses );
				b3FloatW maxImpulse = b3AddW( approachImpulse, b3MaxW( allowance, zeroW ) );
				impulse = b3BlendW( impulse, b3MinW( impulse, maxImpulse ), armed );

				cp->normalImpulses = b3AddW( cp->normalImpulses, impulse );
				cp->restitutionImpulses = b3AddW( cp->restitutionImpulses, b3SubW( impulse, approachImpulse ) );
				cp->totalNormalImpulses = b3AddW( cp->totalNormalImpulses, impulse );

				b3Vec3W P = b3MulSVW( impulse, normal );
				bA.w = b3MulSubMVW( bA.w, c->invIA, b3CrossW( rA, P ) );
				bA.v = b3MulSubSVW( bA.v, c->invMassA, P );
				bB.w = b3MulAddMVW( bB.w, c->invIB, b3CrossW( rB, P ) );
				bB.v = b3MulAddSVW( bB.v, c->invMassB, P );
			}
		}

		b3ScatterBodies( states, c->indexA, &bA );
		b3ScatterBodies( states, c->indexB, &bB );
	}

	b3TracyCZoneEnd( restitution_mesh_wide );
}

void B3_WIDE( b3StoreImpulses_MeshWide )( b3SolverBlock block, b3StepContext* context, int workerIndex )
{
	b3TracyCZoneNC( store_mesh_wide, "Store Mesh", b3_colorFireBrick, true );

	b3World* world = context->world;
	const b3MeshConstraintWide* wideBase = context->wideMeshConstraints;
	b3TaskContext* taskContext = world->taskContexts.data + workerIndex;
	b3BitSet* hitEventBitSet = &taskContext->hitEventBitSet;
	bool hasHitEvents = taskContext->hasHitEvents;
	float negHitThreshold = -world->hitEventThreshold;

	for ( int wideIndex = block.startIndex; wideIndex < block.startIndex + block.count; ++wideIndex )
	{
		const b3MeshConstraintWide* c = wideBase + wideIndex;

		for ( int lane = 0; lane < B3_SIMD_WIDTH; ++lane )
		{
			b3Contact* contact = c->contacts[lane];
			if ( contact == NULL )
			{
				continue;
			}

			int manifoldCount = c->manifoldCounts[lane];
			B3_ASSERT( manifoldCount == contact->manifoldCount );

			bool checkHitEvents = ( contact->flags & b3_simEnableHitEvent ) != 0;
			bool flagged = false;

			for ( int slotIndex = 0; slotIndex < manifoldCount; ++slotIndex )
			{
				const b3MeshManifoldWide* slot = c->manifolds + slotIndex;
				b3Manifold* m = contact->manifolds + slotIndex;

				const float* frictionImpulse1 = (const float*)&slot->frictionImpulse.x;
				const float* frictionImpulse2 = (const float*)&slot->frictionImpulse.y;
				const float* tangent1X = (const float*)&slot->tangent1.X;
				const float* tangent1Y = (const float*)&slot->tangent1.Y;
				const float* tangent1Z = (const float*)&slot->tangent1.Z;
				const float* tangent2X = (const float*)&slot->tangent2.X;
				const float* tangent2Y = (const float*)&slot->tangent2.Y;
				const float* tangent2Z = (const float*)&slot->tangent2.Z;
				const float* twistImpulse = (const float*)&slot->twistImpulse;
				const float* rollingImpulseX = (const float*)&slot->rollingImpulse.X;
				const float* rollingImpulseY = (const float*)&slot->rollingImpulse.Y;
				const float* rollingImpulseZ = (const float*)&slot->rollingImpulse.Z;

				float f1 = frictionImpulse1[lane];
				float f2 = frictionImpulse2[lane];
				m->frictionImpulse = (b3Vec3){
					f1 * tangent1X[lane] + f2 * tangent2X[lane],
					f1 * tangent1Y[lane] + f2 * tangent2Y[lane],
					f1 * tangent1Z[lane] + f2 * tangent2Z[lane],
				};
				m->twistImpulse = twistImpulse[lane];
				m->rollingImpulse = (b3Vec3){
					rollingImpulseX[lane],
					rollingImpulseY[lane],
					rollingImpulseZ[lane],
				};

				int pointCount = m->pointCount;
				B3_ASSERT( pointCount == slot->pointCounts[lane] );
				for ( int pointIndex = 0; pointIndex < pointCount; ++pointIndex )
				{
					const b3ContactConstraintPointWide* cp = slot->points + pointIndex;
					const float* normalImpulse = (const float*)&cp->normalImpulses;
					const float* totalNormalImpulse = (const float*)&cp->totalNormalImpulses;

					b3ManifoldPoint* mp = m->points + pointIndex;
					mp->normalImpulse = normalImpulse[lane];
					mp->totalNormalImpulse = totalNormalImpulse[lane];

					if ( checkHitEvents && flagged == false && mp->normalVelocity < negHitThreshold &&
						 mp->totalNormalImpulse > 0.0f )
					{
						b3SetBit( hitEventBitSet, contact->contactId );
						hasHitEvents = true;
						flagged = true;
					}
				}
			}
		}
	}

	taskContext->hasHitEvents = hasHitEvents;

	b3TracyCZoneEnd( store_mesh_wide );
}
