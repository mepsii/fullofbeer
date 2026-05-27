const container = document.getElementById('container');
const bgCanvas = document.getElementById('bgCanvas');
const beerCanvas = document.getElementById('beerCanvas');
const foamCanvas = document.getElementById('foamCanvas');
const glassCanvas = document.getElementById('glassCanvas');

const bgCtx = bgCanvas.getContext('2d');
const beerCtx = beerCanvas.getContext('2d');
const foamCtx = foamCanvas.getContext('2d');
const glassCtx = glassCanvas.getContext('2d');

// Physics Timestepping Constants (Adjusted for lighter, fast fluid feel)
const SUBSTEPS = 4;             
const GRAVITY = 0.55;           
const H = 12.0;                 // SPH interaction radius
const H_SQ = H * H;
const INV_H = 1.0 / H;          // Precomputed reciprocal of interaction radius
const STIFFNESS = 0.08;         
const STIFFNESS_NEAR = 0.15;    
const REST_DENSITY = 3.0;       

// Cohesive Attraction Parameters (Locking phases together smoothly)
const COHESION = 0.015;         // Attraction between beer particles (capillary cohesion)
const COHESION_FOAM = 0.006;    // Soft attraction between foam particles
const ADHESION = 0.015;         // Subtle adhesive binding at the beer-foam boundary

const VISCOSITY = 0.02;         
const BUOYANCY = 0.40;          // Capped buoyancy for smooth upward transition

// Dynamic Glass Geometry Configuration (recomputed on resize/load)
const GLASS_HEIGHT_DESKTOP = 260;
const R_BASE_DESKTOP = 45;              
const R_TOP_DESKTOP = 70;               
const WALL_THICKNESS = 10;

let currentGlassHeight = GLASS_HEIGHT_DESKTOP;
let currentRBase = R_BASE_DESKTOP;
let currentRTop = R_TOP_DESKTOP;

let width, height;
let isPouring = false;

// Static Memory Allocation (Zero garbage collection latency)
const MAX_PARTICLES = 12000;     // Massive pool limit
class Particle {
    constructor() {
        this.active = false;
        this.x = 0; this.y = 0;
        this.px = 0; this.py = 0;
        this.vx = 0; this.vy = 0;
        this.type = 'beer'; // state of aggregation: 'beer' or 'foam'
        this.liquidType = 'beer'; // beverage identity: 'beer' or 'bourbon'
        this.foamLife = 1.0;
        this.life = 1.0;
        this.inside = false;
        this.insideGlassIndex = -1; // -1 means in the air/floor, >=0 matches glass array
        this.radius = 4;
    }
}
const particlePool = Array.from({ length: MAX_PARTICLES }, () => new Particle());
let activeCount = 0;

// Micro bubbles rising in the glass
const MAX_BUBBLES = 150;
class Bubble {
    constructor() {
        this.active = false;
        this.u = 0; this.v = 0; // Local glass coordinates
        this.vx = 0; this.vy = 0;
        this.radius = 1;
        this.opacity = 1.0;
        this.glassIndex = 0;
    }
}
const bubblePool = Array.from({ length: MAX_BUBBLES }, () => new Bubble());
let activeBubbles = 0;

// Floating Neon Squircle Obstacle Properties
const logo = {
    baseX: 210,
    baseY: 210,
    x: 210,
    y: 210,
    size: 226, // Scaled up by an additional third
    hw: 113,   // half-width
    hh: 113,   // half-height
    r: 45      // corner radius
};

// Offscreen canvas to cache blurred neon bloom
const bloomCanvas = document.createElement('canvas');
const bloomCtx = bloomCanvas.getContext('2d');

function createBloomCache() {
    const pad = 40;
    bloomCanvas.width = logo.size + pad * 2;
    bloomCanvas.height = logo.size + pad * 2;
    
    bloomCtx.clearRect(0, 0, bloomCanvas.width, bloomCanvas.height);
    bloomCtx.save();
    const blurAmount = window.innerWidth <= 768 ? 10 : 18;
    bloomCtx.filter = `blur(${blurAmount}px) saturate(2.0) brightness(1.5)`;
    const innerOffset = 30 * (logo.size / 226);
    bloomCtx.drawImage(logoImg, pad + innerOffset, pad + innerOffset, logo.size - innerOffset * 2, logo.size - innerOffset * 2);
    bloomCtx.restore();
}

const logoImg = new Image();
logoImg.src = 'beercat.png';
let logoLoaded = false;
logoImg.onload = () => { 
    logoLoaded = true; 
    createBloomCache();
};
if (logoImg.complete) {
    logoLoaded = true;
    createBloomCache();
}

// Spatial Hashing Grid buffers
const MAX_GRID_CELLS = 200000;
const grid = new Int32Array(MAX_GRID_CELLS);
const next = new Int32Array(MAX_PARTICLES);
let gridWidth = 0, gridHeight = 0;

// SPH Density Arrays
const densityBuffer = new Float32Array(MAX_PARTICLES);
const nearDensityBuffer = new Float32Array(MAX_PARTICLES);
const foamSubmersion = new Float32Array(MAX_PARTICLES);

// Position-Based Dynamics (PBD) Displacement accumulators
const dispX = new Float32Array(MAX_PARTICLES);
const dispY = new Float32Array(MAX_PARTICLES);

// Spatial Hash precalculated coordinate grids to save math
const gridXBuffer = new Int32Array(MAX_PARTICLES);
const gridYBuffer = new Int32Array(MAX_PARTICLES);

// Pre-allocated cached neighbor pair lists to eliminate redundant searches
const MAX_PAIRS = 250000;
const pairI = new Int32Array(MAX_PAIRS);
const pairJ = new Int32Array(MAX_PAIRS);
const pairDist = new Float32Array(MAX_PAIRS);
const pairQ = new Float32Array(MAX_PAIRS);
const pairRx = new Float32Array(MAX_PAIRS);
const pairRy = new Float32Array(MAX_PAIRS);
let activePairs = 0;

// Pre-allocated viscosity accumulation arrays to avoid garbage collection
const vxAvgBuffer = new Float32Array(MAX_PARTICLES);
const vyAvgBuffer = new Float32Array(MAX_PARTICLES);
const weightSumBuffer = new Float32Array(MAX_PARTICLES);

// Glass array supporting independent instances on desktop
let glasses = [];

const mouse = { x: 0, y: 0, isDown: false };
const keys = { Left: false, Right: false };

// Global precomputed trig values for the glass rotation
let glassCos = 1.0;
let glassSin = 0.0;

function updateGlassAngles() {
    // Keep reference compatible for global calculations
    if (glasses[0]) {
        glassCos = Math.cos(glasses[0].theta);
        glassSin = Math.sin(glasses[0].theta);
    }
}

function localToWorld(u, v) {
    return localToWorldOf(glasses[0], u, v);
}

function worldToLocal(wx, wy) {
    return worldToLocalOf(glasses[0], wx, wy);
}

function localToWorldOf(gl, u, v) {
    if (!gl) return { x: 0, y: 0 };
    const gCos = Math.cos(gl.theta);
    const gSin = Math.sin(gl.theta);
    return {
        x: gl.x + u * gCos + v * gSin,
        y: gl.y + u * gSin - v * gCos
    };
}

function worldToLocalOf(gl, wx, wy) {
    if (!gl) return { u: 0, v: 0 };
    const gCos = Math.cos(gl.theta);
    const gSin = Math.sin(gl.theta);
    const dx = wx - gl.x;
    const dy = wy - gl.y;
    return {
        u: dx * gCos + dy * gSin,
        v: dx * gSin - dy * gCos
    };
}

function updateLogoParams() {
    if (window.innerWidth <= 768) {
        // Mobile layout: smaller, centered, tucked to the top
        logo.size = 120;
        logo.hw = 60;
        logo.hh = 60;
        logo.r = 20;
        logo.baseX = window.innerWidth / 2;
        logo.baseY = 90;
    } else {
        // Desktop layout
        logo.size = 226;
        logo.hw = 113;
        logo.hh = 113;
        logo.r = 45;
        logo.baseX = 210;
        logo.baseY = 210;
    }
    if (logoLoaded) {
        createBloomCache();
    }
}

let shotSpawned = false;

function resize() {
    const oldHeight = height;
    const mobile = window.innerWidth <= 768;

    width = bgCanvas.width = beerCanvas.width = foamCanvas.width = glassCanvas.width = window.innerWidth;
    height = bgCanvas.height = beerCanvas.height = foamCanvas.height = glassCanvas.height = window.innerHeight;
    
    gridWidth = Math.ceil(width / H);
    gridHeight = Math.ceil(height / H);

    currentGlassHeight = mobile ? 120 : GLASS_HEIGHT_DESKTOP;
    currentRBase = mobile ? 22 : R_BASE_DESKTOP;
    currentRTop = mobile ? 24 : R_TOP_DESKTOP;

    const tableY = height - 80;

    if (glasses.length === 0) {
        // Initialize primary glass
        glasses.push({
            type: mobile ? 'shot' : 'pint',
            x: mobile ? window.innerWidth / 2 : 120,
            y: tableY - 27,
            vx: 0, vy: 0, theta: 0, vtheta: 0,
            targetX: mobile ? window.innerWidth / 2 : 120,
            targetY: tableY - 27,
            targetTheta: 0,
            isDragging: false,
            dragOffsetX: 0, dragOffsetY: 0,
            height: currentGlassHeight,
            rBase: currentRBase,
            rTop: currentRTop
        });
    } else {
        glasses[0].type = mobile ? 'shot' : 'pint';
        glasses[0].height = currentGlassHeight;
        glasses[0].rBase = currentRBase;
        glasses[0].rTop = currentRTop;
        
        if (mobile && glasses.length > 1) {
            // Remove desktop extra shot glass on mobile resize
            glasses.splice(1);
            shotSpawned = false;
            const spawnShotBtn = document.getElementById('spawnShotBtn');
            if (spawnShotBtn) {
                spawnShotBtn.innerHTML = '<span class="emoji">🥃</span><span class="btn-text">Spawn Shot</span>';
                spawnShotBtn.style.background = 'rgba(212, 120, 10, 0.25)';
                spawnShotBtn.style.borderColor = 'rgba(212, 120, 10, 0.5)';
            }
        }
    }

    // Move glasses relative to table scaling if necessary
    if (oldHeight !== undefined && oldHeight !== height) {
        const dy = height - oldHeight;
        for (let gIdx = 0; gIdx < glasses.length; gIdx++) {
            glasses[gIdx].y += dy;
            glasses[gIdx].targetY += dy;
        }
        for (let i = 0; i < activeCount; i++) {
            particlePool[i].y += dy;
            particlePool[i].py += dy;
        }
    }

    const spawnShotBtn = document.getElementById('spawnShotBtn');
    if (spawnShotBtn) {
        spawnShotBtn.style.display = mobile ? 'none' : 'flex';
    }

    updateLogoParams();
}
window.addEventListener('resize', resize);

function spawnParticle(x, y, vx, vy, type, inside) {
    spawnParticleInGlass(x, y, vx, vy, type, inside ? 0 : -1);
}

function spawnParticleInGlass(x, y, vx, vy, type, insideGlassIdx, liquidType) {
    if (activeCount < MAX_PARTICLES) {
        const p = particlePool[activeCount];
        p.active = true;
        p.x = x;
        p.y = y;
        p.px = x;
        p.py = y;
        p.vx = vx;
        p.vy = vy;

        // Set dynamic identity context based on source configurations
        const mobile = window.innerWidth <= 768;
        if (liquidType) {
            p.liquidType = liquidType;
        } else {
            if (insideGlassIdx >= 0 && glasses[insideGlassIdx]) {
                p.liquidType = glasses[insideGlassIdx].type === 'shot' ? 'bourbon' : 'beer';
            } else {
                p.liquidType = mobile ? 'bourbon' : 'beer';
            }
        }
        
        // Force strictly to clear liquor 'beer' type if liquid is bourbon (bourbon has no foam!)
        p.type = p.liquidType === 'bourbon' ? 'beer' : type;
        p.foamLife = p.type === 'foam' ? 1.0 : 0.0;
        p.life = 1.0;
        p.insideGlassIndex = insideGlassIdx;
        p.inside = insideGlassIdx >= 0;

        const isShotGlass = p.liquidType === 'bourbon';
        p.radius = p.type === 'beer' ? (isShotGlass ? 2.1 : 4.2) : (isShotGlass ? 2.9 : 5.8);
        activeCount++;
    }
}

function removeParticle(idx) {
    if (idx !== activeCount - 1) {
        const pTarget = particlePool[idx];
        const pLast = particlePool[activeCount - 1];
        
        pTarget.x = pLast.x; pTarget.y = pLast.y;
        pTarget.px = pLast.px; pTarget.py = pLast.py;
        pTarget.vx = pLast.vx; pTarget.vy = pLast.vy;
        pTarget.type = pLast.type;
        pTarget.liquidType = pLast.liquidType; // Preserved during pool recycling array swaps
        pTarget.foamLife = pLast.foamLife;
        pTarget.life = pLast.life;
        pTarget.inside = pLast.inside;
        pTarget.insideGlassIndex = pLast.insideGlassIndex;
        pTarget.radius = pLast.radius;
    }
    activeCount--;
}

function spawnBubble(u, v, gIdx) {
    const gl = glasses[gIdx];
    if (!gl || gl.type === 'shot') return; // Absolutely NO bubbles / carbonation in shot glasses (bourbon)

    if (activeBubbles < MAX_BUBBLES) {
        const b = bubblePool[activeBubbles];
        b.active = true;
        b.u = u;
        b.v = v;
        b.vx = (Math.random() - 0.5) * 0.12;
        b.vy = 0.8 + Math.random() * 1.4;
        b.radius = 0.7 + Math.random() * 1.3;
        b.opacity = 0.3 + Math.random() * 0.5;
        b.glassIndex = gIdx;
        activeBubbles++;
    }
}

function removeBubble(idx) {
    if (idx !== activeBubbles - 1) {
        const bTarget = bubblePool[idx];
        const bLast = bubblePool[activeBubbles - 1];
        bTarget.u = bLast.u; bTarget.v = bLast.v;
        bTarget.vx = bLast.vx; bTarget.vy = bLast.vy;
        bTarget.radius = bLast.radius;
        bTarget.opacity = bLast.opacity;
        bTarget.glassIndex = bLast.glassIndex;
    }
    activeBubbles--;
}

// Signed Distance Field (SDF) of a rounded box (squircle) - Optimized with Math.sqrt
function sdRoundedBox(px, py, cx, cy, hw, hh, r) {
    const dx = Math.abs(px - cx) - (hw - r);
    const dy = Math.abs(py - cy) - (hh - r);
    const mx = Math.max(dx, 0);
    const my = Math.max(dy, 0);
    const distToOuterCorner = Math.sqrt(mx * mx + my * my);
    const distToEdgeInside = Math.min(Math.max(dx, dy), 0);
    return distToOuterCorner + distToEdgeInside - r;
}

function fillSpecificGlass(gIdx) {
    const gl = glasses[gIdx];
    if (!gl) return;

    const floorY = height - 80;
    const R_top_o = gl.rTop + 5; 
    const R_base_o = gl.rBase + 5; 
    const H_g = gl.height;
    const bds = gl.type === 'shot' ? 0.55 : 1.0;

    const boundaryPoints = [
        { u: -R_top_o, v: H_g },                                    
        { u: -(R_base_o + (R_top_o - R_base_o) * 0.66), v: H_g * 0.66 }, 
        { u: -(R_base_o + (R_top_o - R_base_o) * 0.33), v: H_g * 0.33 }, 
        { u: -R_base_o, v: 0 },                                      
        { u: -R_base_o, v: -22 * bds },                                    
        { u: -R_base_o * 0.5, v: -25 * bds },                              
        { u: 0, v: -27 * bds },                                            
        { u: R_base_o * 0.5, v: -25 * bds },                               
        { u: R_base_o, v: -22 * bds },                                     
        { u: R_base_o, v: 0 },                                       
        { u: R_base_o + (R_top_o - R_base_o) * 0.33, v: H_g * 0.33 },  
        { u: R_base_o + (R_top_o - R_base_o) * 0.66, v: H_g * 0.66 },  
        { u: R_top_o, v: H_g }                                       
    ];

    let maxWorldY = -999999;
    const gCos = Math.cos(gl.theta);
    const gSin = Math.sin(gl.theta);

    for (let i = 0; i < boundaryPoints.length; i++) {
        const pt = boundaryPoints[i];
        const wY = gl.y + pt.u * gSin - pt.v * gCos;
        if (wY > maxWorldY) {
            maxWorldY = wY;
        }
    }

    if (maxWorldY > floorY) {
        const penetration = maxWorldY - floorY;
        gl.y -= penetration;
        gl.targetY = gl.y;
    }

    // High-density pre-fill calculations
    // Lower target height (72%) for shot glasses to prevent overlapping pressure spikes
    const fillHeightRatio = gl.type === 'shot' ? 0.72 : 0.96; 
    const targetVMax = gl.height * fillHeightRatio;

    // Wider particle spacing for shot glasses to keep initial packing calm and overflow-free
    const spacingBeer = gl.type === 'shot' ? 2.4 : 3.3; 

    function gLocalToWorld(u, v) {
        return {
            x: gl.x + u * gCos + v * gSin,
            y: gl.y + u * gSin - v * gCos
        };
    }

    // Grid Lattice pre-fill for beer/bourbon
    for (let v = 3.0; v < targetVMax; v += spacingBeer) {
        const halfW = gl.rBase + (gl.rTop - gl.rBase) * (v / gl.height);
        const limit = halfW - (gl.type === 'shot' ? 3.2 : 4.5); // wider boundary margins on shots
        
        const numInRow = Math.floor((limit * 2.0) / spacingBeer);
        if (numInRow > 0) {
            const rowWidth = (numInRow - 1) * spacingBeer;
            const startU = -rowWidth / 2;

            for (let j = 0; j < numInRow; j++) {
                const u = startU + j * spacingBeer;
                if (Math.abs(u) < limit) {
                    const wPos = gLocalToWorld(u, v);
                    spawnParticleInGlass(wPos.x, wPos.y, 0, 0, 'beer', gIdx, gl.type === 'shot' ? 'bourbon' : 'beer');
                }
            }
        }
    }

    // Grid Lattice pre-fill for foam (ONLY generated for pint beer glasses, NOT shot glasses of bourbon)
    if (gl.type !== 'shot') {
        const spacingFoam = 4.8; 
        const foamHeight = 35.0; // Foam layer depth
        for (let v = targetVMax; v < targetVMax + foamHeight; v += spacingFoam) {
            const halfW = gl.rBase + (gl.rTop - gl.rBase) * (v / gl.height);
            const limit = halfW - 6.0; // account for foam particle radius + margin
            
            const numInRow = Math.floor((limit * 2.0) / spacingFoam);
            if (numInRow > 0) {
                const rowWidth = (numInRow - 1) * spacingFoam;
                const startU = -rowWidth / 2;

                for (let j = 0; j < numInRow; j++) {
                    const u = startU + j * spacingFoam;
                    if (Math.abs(u) < limit) {
                        const wPos = gLocalToWorld(u, v);
                        spawnParticleInGlass(wPos.x, wPos.y, 0, 0, 'foam', gIdx, 'beer');
                    }
                }
            }
        }
    }
}

function fillGlass() {
    activeCount = 0;
    activeBubbles = 0;
    for (let gIdx = 0; gIdx < glasses.length; gIdx++) {
        fillSpecificGlass(gIdx);
    }
}

function updatePhysics() {
    const floorY = height - 80;
    const mobile = window.innerWidth <= 768;

    // Ensure dynamic dimensions are mapped correctly
    currentGlassHeight = mobile ? 120 : GLASS_HEIGHT_DESKTOP;
    currentRBase = mobile ? 22 : R_BASE_DESKTOP;
    currentRTop = mobile ? 24 : R_TOP_DESKTOP;

    // Generate drifting bobbing animation for the floating squircle logo (Sway rate lowered)
    const logoTime = Date.now() * 0.0022;
    logo.y = logo.baseY + Math.sin(logoTime) * (mobile ? 1.5 : 3.5);
    logo.x = logo.baseX + Math.cos(logoTime * 0.8) * (mobile ? 0.7 : 1.5);

    for (let gIdx = 0; gIdx < glasses.length; gIdx++) {
        const gl = glasses[gIdx];

        if (mouse.isDown && gl.isDragging) {
            // Keep target coordinates strictly clamped inside canvas viewport borders
            gl.targetX = Math.max(gl.rTop + 10, Math.min(width - gl.rTop - 10, mouse.x - gl.dragOffsetX));
            gl.targetY = Math.max(gl.height + 10, Math.min(floorY - (gl.type === 'shot' ? 15 : 20), mouse.y - gl.dragOffsetY));
            gl.targetTheta = -gl.vx * 0.022;
        } else {
            gl.targetX = Math.max(gl.rTop + 10, Math.min(width - gl.rTop - 10, gl.targetX));
            gl.targetY = Math.min(floorY - (gl.type === 'shot' ? 15 : 20), gl.targetY);
            gl.targetTheta *= 0.88;
        }

        // Tilt keys affect the active glass or the primary glass if nothing else is dragging
        const isTiltActive = gl.isDragging || (glasses.length === 1 && gIdx === 0) || (glasses.length > 1 && gIdx === 0 && !glasses[1].isDragging);
        if (isTiltActive) {
            if (keys.Left) {
                gl.targetTheta = Math.max(-Math.PI * 0.45, gl.targetTheta - 0.05);
            }
            if (keys.Right) {
                gl.targetTheta = Math.min(Math.PI * 0.45, gl.targetTheta + 0.05);
            }
        }

        const targetVx = (gl.targetX - gl.x) * 0.16;
        const targetVy = (gl.targetY - gl.y) * 0.16;
        const targetVtheta = (gl.targetTheta - gl.theta) * 0.22;

        gl.vx += (targetVx - gl.vx) * 0.35;
        gl.vy += (targetVy - gl.vy) * 0.35;
        gl.vtheta += (targetVtheta - gl.vtheta) * 0.35;
    }

    const dt = 1.0 / SUBSTEPS;

    for (let step = 0; step < SUBSTEPS; step++) {
        
        // Continuous substepped spigot pouring
        if (isPouring) {
            const tapX = width - (mobile ? 60 : 150); // Align spawn physics
            
            // Context-Aware Pouring: Determine liquid based on which glass is positioned under the tap
            let pourLiquidType = mobile ? 'bourbon' : 'beer';
            if (!mobile) {
                for (let gIdx = 0; gIdx < glasses.length; gIdx++) {
                    const gl = glasses[gIdx];
                    if (Math.abs(gl.x - tapX) < 100) {
                        pourLiquidType = gl.type === 'shot' ? 'bourbon' : 'beer';
                        break;
                    }
                }
            }

            for (let i = 0; i < 2; i++) { 
                const spawnX = tapX + (Math.random() - 0.5) * (mobile ? 1.5 : 3);
                const spawnY = (mobile ? 35 : 55) + i * (mobile ? 2.5 : 5.5);
                const vx = 0;   
                const vy = mobile ? 6.0 : 9.0;  // slower velocity for shot glass to prevent splash overshoot
                const type = Math.random() < (mobile ? 0.01 : 0.04) ? 'foam' : 'beer';
                spawnParticleInGlass(spawnX, spawnY, vx, vy, type, -1, pourLiquidType);
            }
        }

        // Increment glass coordinates
        for (let gIdx = 0; gIdx < glasses.length; gIdx++) {
            const gl = glasses[gIdx];
            gl.x += gl.vx * dt;
            gl.y += gl.vy * dt;
            gl.theta += gl.vtheta * dt;
        }

        // Precompute dynamic trig structures for this physics substep
        updateGlassAngles();

        // --- Glass-to-Table collision resolution (Check all glasses) ---
        for (let gIdx = 0; gIdx < glasses.length; gIdx++) {
            const gl = glasses[gIdx];
            const gCos = Math.cos(gl.theta);
            const gSin = Math.sin(gl.theta);

            const R_top_o = gl.rTop + 5; 
            const R_base_o = gl.rBase + 5; 
            const H_g = gl.height;
            const bds = gl.type === 'shot' ? 0.55 : 1.0;

            const boundaryPoints = [
                { u: -R_top_o, v: H_g },                                    
                { u: -(R_base_o + (R_top_o - R_base_o) * 0.66), v: H_g * 0.66 }, 
                { u: -(R_base_o + (R_top_o - R_base_o) * 0.33), v: H_g * 0.33 }, 
                { u: -R_base_o, v: 0 },                                      
                { u: -R_base_o, v: -22 * bds },                                    
                { u: -R_base_o * 0.5, v: -25 * bds },                              
                { u: 0, v: -27 * bds },                                            
                { u: R_base_o * 0.5, v: -25 * bds },                               
                { u: R_base_o, v: -22 * bds },                                     
                { u: R_base_o, v: 0 },                                       
                { u: R_base_o + (R_top_o - R_base_o) * 0.33, v: H_g * 0.33 },  
                { u: R_base_o + (R_top_o - R_base_o) * 0.66, v: H_g * 0.66 },  
                { u: R_top_o, v: H_g }                                       
            ];

            let maxWorldY = -999999;
            let lowestU = 0;
            let lowestV = 0;

            for (let i = 0; i < boundaryPoints.length; i++) {
                const pt = boundaryPoints[i];
                const wY = gl.y + pt.u * gSin - pt.v * gCos;
                if (wY > maxWorldY) {
                    maxWorldY = wY;
                    lowestU = pt.u;
                    lowestV = pt.v;
                }
            }

            if (maxWorldY > floorY) {
                const penetration = maxWorldY - floorY;
                gl.y -= penetration; // Positional correction to prevent clipping

                const rx = lowestU * gCos + lowestV * gSin;
                const relVelY = gl.vy + gl.vtheta * rx;

                if (relVelY > 0) {
                    const restitution = 0.35; // bounce intensity
                    const mass = 1.0;
                    const inertia = gl.type === 'shot' ? 800.0 : 2500.0; // scale inertia based on glass mass
                    
                    const impulse = (1.0 + restitution) * relVelY / (1.0 / mass + (rx * rx) / inertia);

                    gl.vy -= impulse / mass;
                    gl.vtheta -= (impulse * rx) / inertia;
                    gl.vx *= 0.82; // horizontal friction on impact
                }
            }

            // --- Glass-to-Obstacle (Logo) Collision solver ---
            let maxLogoPen = 0;
            let colPtLogoX = 0;
            let colPtLogoY = 0;
            let colNormalLogoX = 0;
            let colNormalLogoY = 0;
            const glassThicknessMargin = gl.type === 'shot' ? 2.5 : 5.0; // Margin matching physical glass bounds

            for (let i = 0; i < boundaryPoints.length; i++) {
                const pt = boundaryPoints[i];
                const wX = gl.x + pt.u * gCos + pt.v * gSin;
                const wY = gl.y + pt.u * gSin - pt.v * gCos;

                const sd = sdRoundedBox(wX, wY, logo.x, logo.y, logo.hw, logo.hh, logo.r);
                if (sd < glassThicknessMargin) {
                    const pen = glassThicknessMargin - sd;
                    if (pen > maxLogoPen) {
                        maxLogoPen = pen;
                        colPtLogoX = wX;
                        colPtLogoY = wY;

                        // Normal calculation via central difference approximations on SDF
                        const eps = 0.5;
                        const dX = sdRoundedBox(wX + eps, wY, logo.x, logo.y, logo.hw, logo.hh, logo.r) - sdRoundedBox(wX - eps, wY, logo.x, logo.y, logo.hw, logo.hh, logo.r);
                        const dY = sdRoundedBox(wX, wY + eps, logo.x, logo.y, logo.hw, logo.hh, logo.r) - sdRoundedBox(wX, wY - eps, logo.x, logo.y, logo.hw, logo.hh, logo.r);
                        const len = Math.sqrt(dX * dX + dY * dY) || 0.001;
                        colNormalLogoX = dX / len;
                        colNormalLogoY = dY / len;
                    }
                }
            }

            if (maxLogoPen > 0) {
                // Correct overlapping coordinates instantly
                gl.x += colNormalLogoX * maxLogoPen;
                gl.y += colNormalLogoY * maxLogoPen;

                const rx = colPtLogoX - gl.x;
                const ry = colPtLogoY - gl.y;

                // Normal linear contact point velocity
                const vcX = gl.vx - gl.vtheta * ry;
                const vcY = gl.vy + gl.vtheta * rx;
                const normalVel = vcX * colNormalLogoX + vcY * colNormalLogoY;

                if (normalVel < 0) {
                    const restitution = 0.45; // Springy neon sign bounce reaction
                    const mass = 1.0;
                    const inertia = gl.type === 'shot' ? 800.0 : 2500.0;

                    const r_cross_n = rx * colNormalLogoY - ry * colNormalLogoX;
                    const impulse = -(1.0 + restitution) * normalVel / (1.0 / mass + (r_cross_n * r_cross_n) / inertia);

                    // Adjust kinetic vectors instantly
                    gl.vx += (impulse / mass) * colNormalLogoX;
                    gl.vy += (impulse / mass) * colNormalLogoY;
                    gl.vtheta += (impulse * r_cross_n) / inertia;

                    // Tangential sliding friction damping
                    const tangX = -colNormalLogoY;
                    const tangY = colNormalLogoX;
                    const tangVel = vcX * tangX + vcY * tangY;
                    gl.vx -= tangVel * 0.15 * tangX;
                    gl.vy -= tangVel * 0.15 * tangY;
                }
            }
        }

        // Dynamic Pool Recycling: Clear outside particles if approaching performance limits
        if (activeCount > MAX_PARTICLES - 200) {
            for (let i = activeCount - 1; i >= 0; i--) {
                if (particlePool[i].insideGlassIndex < 0) {
                    removeParticle(i);
                    if (activeCount <= MAX_PARTICLES - 500) break;
                }
            }
        }

        // Step 1: External Forces, Viscosity, and Buoyancy updates
        for (let i = 0; i < activeCount; i++) {
            const p = particlePool[i];

            // Standard gravity is applied equally to both beer and foam
            p.vy += GRAVITY * dt;

            const insideGlass = p.insideGlassIndex >= 0 ? glasses[p.insideGlassIndex] : null;

            if (p.type === 'foam') {
                // Buoyancy ONLY acts inside the glass (fluid columns)
                if (insideGlass) {
                    const buoyForce = Math.min(1.2, BUOYANCY * foamSubmersion[i]);
                    p.vy -= buoyForce * dt;
                }
            }

            // HYBRID DAMPING: Damps velocity RELATIVE to the moving glass context inside.
            if (insideGlass) {
                let relVx = p.vx - insideGlass.vx;
                let relVy = p.vy - insideGlass.vy;
                
                // Foam has higher relative damping to quickly freeze vibration; beer is lively but settles.
                let drag = p.type === 'foam' ? 0.15 : 0.025; 
                
                relVx *= (1.0 - drag * dt);
                relVy *= (1.0 - drag * dt);
                
                p.vx = insideGlass.vx + relVx;
                p.vy = insideGlass.vy + relVy;
            } else {
                // Standard, minimal world-space drag in the air (natural ballistic weight)
                p.vx *= (1.0 - 0.012 * dt);
                p.vy *= (1.0 - 0.012 * dt);
            }

            p.px = p.x;
            p.py = p.y;
            p.x += p.vx * dt;
            p.y += p.vy * dt;
        }

        // Step 2: Build Spatial Hashing Grid (Pre-cache particle coordinates)
        const numCells = gridWidth * gridHeight;
        if (numCells < MAX_GRID_CELLS) {
            grid.subarray(0, numCells).fill(-1);
        } else {
            grid.fill(-1);
        }

        for (let i = 0; i < activeCount; i++) {
            const p = particlePool[i];
            const gx = Math.floor(p.x * INV_H);
            const gy = Math.floor(p.y * INV_H);
            gridXBuffer[i] = gx;
            gridYBuffer[i] = gy;
            if (gx >= 0 && gx < gridWidth && gy >= 0 && gy < gridHeight) {
                const cellIdx = gx + gy * gridWidth;
                next[i] = grid[cellIdx];
                grid[cellIdx] = i;
            } else {
                next[i] = -1;
            }
        }

        // Step 3: Calculate Local SPH Densities & Build Cached Pair List (Saves 50% loop searches)
        densityBuffer.subarray(0, activeCount).fill(0);
        nearDensityBuffer.subarray(0, activeCount).fill(0);
        foamSubmersion.subarray(0, activeCount).fill(0);
        activePairs = 0;

        for (let i = 0; i < activeCount; i++) {
            const p = particlePool[i];
            const gx = gridXBuffer[i];
            const gy = gridYBuffer[i];
            
            for (let dy = -1; dy <= 1; dy++) {
                const ny = gy + dy;
                if (ny < 0 || ny >= gridHeight) continue;
                const rowOffset = ny * gridWidth;
                for (let dx = -1; dx <= 1; dx++) {
                    const nx = gx + dx;
                    if (nx < 0 || nx >= gridWidth) continue;
                    const cellIdx = nx + rowOffset;
                    let j = grid[cellIdx];
                    while (j !== -1) {
                        if (j > i) {
                            const pj = particlePool[j];
                            const rx = pj.x - p.x;
                            const ry = pj.y - p.y;
                            const distSq = rx * rx + ry * ry;
                            if (distSq < H_SQ) {
                                const dist = Math.sqrt(distSq) || 0.1;
                                const q = 1.0 - (dist * INV_H);
                                const q2 = q * q;
                                const q3 = q2 * q;

                                densityBuffer[i] += q2;
                                densityBuffer[j] += q2;

                                nearDensityBuffer[i] += q3;
                                nearDensityBuffer[j] += q3;

                                if (p.type === 'foam' && pj.type === 'beer') {
                                    foamSubmersion[i] += q2;
                                } else if (pj.type === 'foam' && p.type === 'beer') {
                                    foamSubmersion[j] += q2;
                                }

                                // Store interacting pair parameters to bypass grid search in Steps 4 and 5
                                if (activePairs < MAX_PAIRS) {
                                    pairI[activePairs] = i;
                                    pairJ[activePairs] = j;
                                    pairDist[activePairs] = dist;
                                    pairQ[activePairs] = q;
                                    pairRx[activePairs] = rx;
                                    pairRy[activePairs] = ry;
                                    activePairs++;
                                }
                            }
                        }
                        j = next[j];
                    }
                }
            }
        }

        // Step 4: SPH Relaxation & Displacement (Bypasses grid traversal entirely via cached pairs)
        dispX.fill(0);
        dispY.fill(0);

        for (let idx = 0; idx < activePairs; idx++) {
            const i = pairI[idx];
            const j = pairJ[idx];
            const dist = pairDist[idx];
            const q = pairQ[idx];
            const rx = pairRx[idx];
            const ry = pairRy[idx];

            const p = particlePool[i];
            const pj = particlePool[j];

            const press = STIFFNESS * (densityBuffer[i] - REST_DENSITY);
            const pressNear = STIFFNESS_NEAR * nearDensityBuffer[i];
            
            let force = (press + pressNear * q) * q * dt * dt;
            
            // Soften SPH repulsion for foam interactions (prevents clumping/harsh separation)
            if (p.type === 'foam' && pj.type === 'foam') {
                force *= 0.15; 
            } else if ((p.type === 'foam' && pj.type === 'beer') || (p.type === 'beer' && pj.type === 'foam')) {
                force *= 0.30;
            }

            // Cohesive Phase-Locking interactions (Glues foam head to beer surface)
            if (p.type === 'beer' && pj.type === 'beer') {
                force -= COHESION * q * q * dt * dt;
            } else if (p.type === 'foam' && pj.type === 'foam') {
                force -= COHESION_FOAM * q * q * dt * dt;
            } else {
                force -= ADHESION * q * q * dt * dt; // Soft Adhesion
            }

            force = Math.max(-0.6, Math.min(0.6, force));

            // Micro-optimized reciprocal division
            const fOverDist = force / dist;
            const fx = rx * fOverDist;
            const fy = ry * fOverDist;

            dispX[i] -= fx;
            dispY[i] -= fy;
            dispX[j] += fx;
            dispY[j] += fy;
        }

        // Apply accumulated displacements
        for (let i = 0; i < activeCount; i++) {
            const p = particlePool[i];
            p.x += dispX[i];
            p.y += dispY[i];
        }

        // Step 5: Smooth Viscosity step (XSPH - Bypasses grid traversal via cached pairs)
        vxAvgBuffer.subarray(0, activeCount).fill(0);
        vyAvgBuffer.subarray(0, activeCount).fill(0);
        weightSumBuffer.subarray(0, activeCount).fill(0);

        for (let idx = 0; idx < activePairs; idx++) {
            const i = pairI[idx];
            const j = pairJ[idx];
            const q = pairQ[idx];

            const p = particlePool[i];
            const pj = particlePool[j];

            vxAvgBuffer[i] += pj.vx * q;
            vyAvgBuffer[i] += pj.vy * q;
            weightSumBuffer[i] += q;

            vxAvgBuffer[j] += p.vx * q;
            vyAvgBuffer[j] += p.vy * q;
            weightSumBuffer[j] += q;
        }

        for (let i = 0; i < activeCount; i++) {
            const p = particlePool[i];
            const weightSum = weightSumBuffer[i];
            if (weightSum > 0) {
                const vxAvg = vxAvgBuffer[i] / weightSum;
                const vyAvg = vyAvgBuffer[i] / weightSum;
                
                // Relative internal shear damping (viscosity) makes foam clump cream-like
                const vFactor = p.type === 'foam' ? 0.25 : VISCOSITY;
                p.vx += (vxAvg - p.vx) * vFactor * dt;
                p.vy += (vyAvg - p.vy) * vFactor * dt;
            }
        }

        // Find local liquid height of each glass to pop rising bubbles
        let maxLiquidVArray = new Float32Array(glasses.length).fill(10);
        for (let i = 0; i < activeCount; i++) {
            const p = particlePool[i];
            if (p.insideGlassIndex >= 0) {
                const gl = glasses[p.insideGlassIndex];
                if (gl) {
                    const loc = worldToLocalOf(gl, p.x, p.y);
                    if (loc.v > maxLiquidVArray[p.insideGlassIndex]) {
                        maxLiquidVArray[p.insideGlassIndex] = loc.v;
                    }
                }
            }
        }

        // Step 6: Resolve Glass & Solid Boundary Collisions
        for (let i = activeCount - 1; i >= 0; i--) {
            const p = particlePool[i];

            // Floor Collision
            if (p.y > floorY) {
                p.y = floorY;
                if (p.insideGlassIndex < 0) {
                    // Splattered outside particles delete instantly on floor impact
                    removeParticle(i);
                    continue;
                }
                p.vy = -p.vy * 0.1;
                p.vx *= 0.5;
            }

            // Obstacle Collision solver using dynamic squircle distance fields (SDF) with fast AABB pre-filter
            const colDist = p.radius + 1.2; // Add soft clearance layer
            const distX = Math.abs(p.x - logo.x);
            const distY = Math.abs(p.y - logo.y);
            const margin = logo.hw + colDist + 5; // Bounding margin around squircle

            if (distX < margin && distY < margin) {
                const sd = sdRoundedBox(p.x, p.y, logo.x, logo.y, logo.hw, logo.hh, logo.r);
                if (sd < colDist) {
                    const penetration = colDist - sd;
                    
                    // Finite difference numerical boundary normal calculation
                    const eps = 0.5;
                    const dX = sdRoundedBox(p.x + eps, p.y, logo.x, logo.y, logo.hw, logo.hh, logo.r) - sdRoundedBox(p.x - eps, p.y, logo.x, logo.y, logo.hw, logo.hh, logo.r);
                    const dY = sdRoundedBox(p.x, p.y + eps, logo.x, logo.y, logo.hw, logo.hh, logo.r) - sdRoundedBox(p.x, p.y - eps, logo.x, logo.y, logo.hw, logo.hh, logo.r);
                    const len = Math.sqrt(dX * dX + dY * dY) || 0.001;
                    const invLen = 1.0 / len;
                    const nx = dX * invLen;
                    const ny = dY * invLen;

                    p.x += nx * penetration;
                    p.y += ny * penetration;

                    const dot = p.vx * nx + p.vy * ny;
                    if (dot < 0) {
                        // Elastic bounce + friction
                        p.vx -= 1.35 * dot * nx;
                        p.vy -= 1.35 * dot * ny;
                        p.vx *= 0.85; 
                        p.vy *= 0.85;

                        // Turn high velocity impacts into foam splashes (ONLY if NOT shot glass and NOT bourbon particle)
                        const insideGlassType = p.insideGlassIndex >= 0 ? glasses[p.insideGlassIndex].type : 'pint';
                        if (insideGlassType !== 'shot' && p.liquidType !== 'bourbon' && p.type === 'beer' && Math.abs(dot) > 4.0 && Math.random() < 0.25) {
                            p.type = 'foam';
                            p.radius = mobile ? 2.9 : 5.8;
                            p.foamLife = 0.4 + Math.random() * 0.4;
                        }
                    }
                }
            }

            // Recapture Falling Stream (Check all glasses)
            if (p.insideGlassIndex < 0) {
                for (let gIdx = 0; gIdx < glasses.length; gIdx++) {
                    const gl = glasses[gIdx];
                    let loc = worldToLocalOf(gl, p.x, p.y);
                    if (loc.v > 0 && loc.v < gl.height) {
                        const halfW = gl.rBase + (gl.rTop - gl.rBase) * (loc.v / gl.height);
                        if (Math.abs(loc.u) < halfW - 2) {
                            p.insideGlassIndex = gIdx;
                            p.inside = true;
                            if (gl.type === 'shot') {
                                p.type = 'beer'; // strictly prevent froth formation inside shots
                            }
                            break;
                        }
                    }
                }
            }

            // Inside Glass Walls Resolution
            if (p.insideGlassIndex >= 0) {
                const gl = glasses[p.insideGlassIndex];
                if (!gl) {
                    p.insideGlassIndex = -1;
                    p.inside = false;
                    continue;
                }

                let loc = worldToLocalOf(gl, p.x, p.y);
                let collided = false;

                // Cavity floor
                if (loc.v < 0) {
                    loc.v = 0;
                    collided = true;
                }

                // Slanted cylinder walls
                if (loc.v >= 0 && loc.v <= gl.height) {
                    const halfW = gl.rBase + (gl.rTop - gl.rBase) * (loc.v / gl.height);
                    const limit = halfW - p.radius * 0.45;
                    if (Math.abs(loc.u) > limit) {
                        loc.u = Math.sign(loc.u) * limit;
                        collided = true;
                    }
                }

                // Mouth/rim escape check 
                if (loc.v > gl.height) {
                    if (Math.abs(loc.u) > gl.rTop) {
                        p.insideGlassIndex = -1; // Spilled over lip
                        p.inside = false;
                    }
                }

                // Splash-to-foam transition (ONLY if NOT a shot glass and NOT bourbon particle)
                if (collided && p.type === 'beer' && gl.type !== 'shot' && p.liquidType !== 'bourbon') {
                    const speed = Math.sqrt(p.vx * p.vx + p.vy * p.vy);
                    if (speed > 7.0 && Math.random() < 0.04) {
                        p.type = 'foam';
                        p.radius = mobile ? 2.9 : 5.8;
                        p.foamLife = 0.5 + Math.random() * 0.5;
                    }
                }

                if (collided) {
                    const worldPos = localToWorldOf(gl, loc.u, loc.v);
                    p.x = worldPos.x;
                    p.y = worldPos.y;
                }
            } else {
                // Sliding outside wall collisions (Check all glasses)
                for (let gIdx = 0; gIdx < glasses.length; gIdx++) {
                    const gl = glasses[gIdx];
                    let loc = worldToLocalOf(gl, p.x, p.y);
                    const outerDepth = gl.type === 'shot' ? -14 : -25;
                    const R_outer = gl.rBase + (gl.rTop - gl.rBase) * (Math.max(0, Math.min(gl.height, loc.v)) / gl.height) + WALL_THICKNESS;
                    if (loc.v >= outerDepth && loc.v <= gl.height && Math.abs(loc.u) < R_outer) {
                        const distL = Math.abs(loc.u + R_outer);
                        const distR = Math.abs(loc.u - R_outer);
                        const distB = Math.abs(loc.v - outerDepth);
                        const minDist = Math.min(distL, distR, distB);

                        if (minDist === distL) {
                            loc.u = -R_outer;
                        } else if (minDist === distR) {
                            loc.u = R_outer;
                        } else {
                            loc.v = outerDepth;
                        }
                        const worldPos = localToWorldOf(gl, loc.u, loc.v);
                        p.x = worldPos.x;
                        p.y = worldPos.y;
                        break; // Handled collision
                    }
                }
            }

            p.vx = (p.x - p.px) / dt;
            p.vy = (p.y - p.py) / dt;

            const speed = Math.sqrt(p.vx * p.vx + p.vy * p.vy);
            if (speed > 16.0) {
                const invSpeed = 16.0 / speed;
                p.vx *= invSpeed;
                p.vy *= invSpeed;
            }

            const insideGlassType = p.insideGlassIndex >= 0 ? glasses[p.insideGlassIndex].type : 'pint';
            const idealRadius = p.type === 'beer' ? (p.liquidType === 'bourbon' ? 2.1 : 4.2) : (p.liquidType === 'bourbon' ? 2.9 : 5.8);
            p.radius = idealRadius;

            // Absolute removal of foam inside shot glasses or for bourbon particles (bourbon is completely froth-free)
            if ((insideGlassType === 'shot' || p.liquidType === 'bourbon') && p.type === 'foam') {
                p.type = 'beer';
                p.radius = 2.1;
            }

            // Foam aging and dissolving (Only 1/3 equivalent volume turns to beer; 2/3 vanishes)
            if (p.type === 'foam' && p.insideGlassIndex >= 0) {
                p.foamLife -= 0.0035 * dt;
                if (p.foamLife <= 0) {
                    if (Math.random() < 0.33) {
                        p.type = 'beer';
                        p.radius = insideGlassType === 'shot' ? 2.1 : 4.2;
                    } else {
                        removeParticle(i);
                        continue;
                    }
                }
            }
        }

        // Step 7: Update Micro Bubble physics (rising vertically inside)
        for (let i = activeBubbles - 1; i >= 0; i--) {
            const b = bubblePool[i];
            const gl = glasses[b.glassIndex];
            if (!gl || gl.type === 'shot') {
                // Pop bubbles immediately inside shot glasses (Smooth bourbon has no fizz)
                removeBubble(i);
                continue;
            }

            b.u += b.vx;
            b.v += b.vy;
            b.vx += (Math.random() - 0.5) * 0.04;
            b.vx *= 0.95;

            const halfW = gl.rBase + (gl.rTop - gl.rBase) * (b.v / gl.height) - 3;
            if (Math.abs(b.u) > halfW) {
                b.u = Math.sign(b.u) * halfW;
                b.vx *= -0.5;
            }

            // Pop bubbles at the dynamic liquid surface level or glass rim
            const maxLiquidV = maxLiquidVArray[b.glassIndex];
            if (b.v > maxLiquidV || b.v > gl.height - 8) {
                // Spawns a tiny foam particle on the surface when bubble pops
                if (Math.random() < 0.25 && b.v < gl.height - 12 && activeCount < MAX_PARTICLES) {
                    const wPos = localToWorldOf(gl, b.u, b.v);
                    spawnParticleInGlass(wPos.x, wPos.y, 0, -0.2, 'foam', b.glassIndex);
                }
                removeBubble(i);
            }
        }

        // Continual carbonation fizz nucleation from physical glass bottom (Only for non-shot glasses!)
        for (let gIdx = 0; gIdx < glasses.length; gIdx++) {
            const gl = glasses[gIdx];
            if (gl.type !== 'shot') {
                if (Math.random() < 0.35 && activeBubbles < MAX_BUBBLES) {
                    const u = (Math.random() - 0.5) * gl.rBase * 1.5;
                    const v = 2 + Math.random() * 8;
                    spawnBubble(u, v, gIdx);
                }
            }
        }
    }
}

// Custom path generator for rendering squircles
function drawSquircle(ctx, x, y, size, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + size - r, y);
    ctx.quadraticCurveTo(x + size, y, x + size, y + r);
    ctx.lineTo(x + size, y + size - r);
    ctx.quadraticCurveTo(x + size, y + size, x + size - r, y + size);
    ctx.lineTo(x + r, y + size);
    ctx.quadraticCurveTo(x, y + size, x, y + size - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
}

// Custom high-quality vector neon glow stroke function
function drawNeonSquircle(ctx, x, y, size, r, colorOuter, colorInner) {
    const sf = size / 194; // Exactly 1 on desktop (226 - 32)
    ctx.save();
    
    // Pass 1: Saturated outer volumetric glow
    ctx.shadowColor = colorOuter;
    ctx.shadowBlur = 35 * sf;
    ctx.strokeStyle = colorOuter;
    ctx.lineWidth = 12 * sf;
    ctx.lineJoin = 'round';
    drawSquircle(ctx, x, y, size, r);
    ctx.stroke();

    // Pass 2: Hot core discharge glow
    ctx.shadowBlur = 15 * sf;
    ctx.strokeStyle = colorInner;
    ctx.lineWidth = 6 * sf;
    ctx.stroke();

    // Pass 3: White-hot central path
    ctx.shadowBlur = 4 * sf;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2.5 * sf;
    ctx.stroke();

    ctx.restore();
}

function draw() {
    bgCtx.clearRect(0, 0, width, height);
    beerCtx.clearRect(0, 0, width, height);
    foamCtx.clearRect(0, 0, width, height);
    glassCtx.clearRect(0, 0, width, height);

    const floorY = height - 80;
    const mobile = window.innerWidth <= 768;

    // Draw Wooden Table Top
    bgCtx.fillStyle = '#1c0e07';
    bgCtx.fillRect(0, floorY, width, 80);
    bgCtx.fillStyle = '#120905';
    bgCtx.fillRect(0, floorY, width, 8);

    // Draw Brass Tap Spigot Nozzle in the Right Corner
    const tapX = width - (mobile ? 60 : 150);
    bgCtx.fillStyle = '#cda142'; 
    bgCtx.fillRect(tapX - (mobile ? 30 : 60), 0, (mobile ? 60 : 120), 20); 
    bgCtx.fillRect(tapX - (mobile ? 6 : 12), 20, (mobile ? 12 : 24), 38); 
    bgCtx.fillStyle = '#1a0d07'; 
    bgCtx.fillRect(tapX - (mobile ? 3 : 6), 56, (mobile ? 6 : 12), 4);

    // Tap mechanical lever
    bgCtx.strokeStyle = '#cda142';
    bgCtx.lineWidth = mobile ? 3 : 6;
    bgCtx.beginPath();
    bgCtx.moveTo(tapX, 15);
    const leverAngle = isPouring ? 0.38 : -0.38;
    const leverLen = mobile ? 22 : 45;
    bgCtx.lineTo(tapX + Math.sin(leverAngle) * leverLen, 15 - Math.cos(leverAngle) * leverLen);
    bgCtx.stroke();
    
    // Handle Knob
    bgCtx.fillStyle = '#111';
    bgCtx.beginPath();
    bgCtx.arc(tapX + Math.sin(leverAngle) * leverLen, 15 - Math.cos(leverAngle) * leverLen, mobile ? 4 : 8, 0, Math.PI * 2);
    bgCtx.fill();

    // SINGLE-PATH GPU BATCH RENDERING (Uses squares 'rect' to bypass complex curves, blurred automatically by SVG)
    
    // 1. Draw clear golden beer (for pint glass or in the air on desktop)
    beerCtx.fillStyle = '#ffb300';
    beerCtx.beginPath();
    for (let i = 0; i < activeCount; i++) {
        const p = particlePool[i];
        if (p.type === 'beer' && p.liquidType === 'beer') {
            const r = p.radius;
            beerCtx.rect(p.x - r, p.y - r, r * 2, r * 2);
        }
    }
    beerCtx.fill();

    // 1b. Draw rich heavy bourbon (Always stays dark, even when spilled outside the glass!)
    beerCtx.fillStyle = '#5c2c06';
    beerCtx.beginPath();
    for (let i = 0; i < activeCount; i++) {
        const p = particlePool[i];
        if (p.type === 'beer' && p.liquidType === 'bourbon') {
            const r = p.radius;
            beerCtx.rect(p.x - r, p.y - r, r * 2, r * 2);
        }
    }
    beerCtx.fill();

    // 2. Draw white beer foam
    foamCtx.fillStyle = '#faf6f0';
    foamCtx.beginPath();
    for (let i = 0; i < activeCount; i++) {
        const p = particlePool[i];
        if (p.type === 'foam' && p.liquidType === 'beer') {
            const r = p.radius;
            foamCtx.rect(p.x - r, p.y - r, r * 2, r * 2);
        }
    }
    foamCtx.fill();

    // 2b. Draw bourbon foam/splash particles (translucent amber) - absolutely no froth is drawn inside shot glasses
    foamCtx.fillStyle = 'rgba(210, 105, 30, 0.45)';
    foamCtx.beginPath();
    for (let i = 0; i < activeCount; i++) {
        const p = particlePool[i];
        if (p.type === 'foam' && p.liquidType === 'bourbon') {
            const r = p.radius;
            foamCtx.rect(p.x - r, p.y - r, r * 2, r * 2);
        }
    }
    foamCtx.fill();

    // Render Floating Squircle Sign (On glass canvas for overlay visibility)
    glassCtx.save();
    
    // 1. Wires hanging down from the ceiling (spaced to match scaled size)
    glassCtx.strokeStyle = '#1e1e1e';
    glassCtx.lineWidth = mobile ? 1.5 : 3;
    glassCtx.beginPath();
    // Left Cable
    glassCtx.moveTo(logo.x - (mobile ? 35 : 70), 0);
    glassCtx.lineTo(logo.x - (mobile ? 35 : 70), logo.y - logo.size / 2 + 10);
    // Right Cable
    glassCtx.moveTo(logo.x + (mobile ? 35 : 70), 0);
    glassCtx.lineTo(logo.x + (mobile ? 35 : 70), logo.y - logo.size / 2 + 10);
    glassCtx.stroke();

    // 2. Heavy Sign Shadow & Chassis Plate
    glassCtx.shadowColor = 'rgba(0, 0, 0, 0.75)';
    glassCtx.shadowBlur = mobile ? 12 : 24;
    glassCtx.shadowOffsetY = mobile ? 6 : 12;

    const sf = logo.size / 226; // Desktop reference size is 226
    const logoX = logo.x - logo.size / 2;
    const logoY = logo.y - logo.size / 2;
    const logoOuterRadius = logo.r + 5 * sf;

    // Matte-black obsidian chassis backing
    glassCtx.fillStyle = 'rgba(12, 12, 12, 0.94)';
    glassCtx.lineWidth = 6 * sf;
    glassCtx.strokeStyle = '#222222';
    drawSquircle(glassCtx, logoX, logoY, logo.size, logoOuterRadius);
    glassCtx.fill();
    glassCtx.stroke();

    // Disable standard blur for inner details
    glassCtx.shadowBlur = 0;
    glassCtx.shadowOffsetY = 0;

    // Inner framing accent bezel
    glassCtx.lineWidth = 2.5 * sf;
    glassCtx.strokeStyle = '#3a3a3a';
    drawSquircle(glassCtx, logoX + 5 * sf, logoY + 5 * sf, logo.size - 10 * sf, logoOuterRadius - 2 * sf);
    glassCtx.stroke();

    // 3. Render Neon Border (Deep Orange / Amber glow theme)
    drawNeonSquircle(glassCtx, logoX + 16 * sf, logoY + 16 * sf, logo.size - 32 * sf, logoOuterRadius - 8 * sf, '#ff3c00', '#ff9f3b');

    // 4. Render organic color-sensitive "bloom" glow (Optimized offscreen cache draws in < 1ms)
    if (logoLoaded) {
        glassCtx.save();
        glassCtx.globalCompositeOperation = 'screen'; // Organic additive light blend
        const pad = 40;
        glassCtx.drawImage(bloomCanvas, logoX - pad, logoY - pad);
        glassCtx.restore();
    }

    // 5. Clip area inside neon boundary to map sharp logo graphic
    glassCtx.save();
    drawSquircle(glassCtx, logoX + 30 * sf, logoY + 30 * sf, logo.size - 60 * sf, logoOuterRadius - 16 * sf);
    glassCtx.clip();

    if (logoLoaded) {
        // Draw custom logo asset inside glowing window
        glassCtx.drawImage(logoImg, logoX + 30 * sf, logoY + 30 * sf, logo.size - 60 * sf, logo.size - 60 * sf);
    } else {
        // Custom glowing vector fallback artwork with matched organic neon emission
        glassCtx.fillStyle = '#1c1c1c';
        glassCtx.fillRect(logoX, logoY, logo.size, logo.size);

        glassCtx.shadowColor = '#ff6c00';
        glassCtx.shadowBlur = 25 * sf; // Large glowing fallback bloom
        glassCtx.fillStyle = '#ff6c00';
        glassCtx.font = `bold ${Math.round(50 * sf)}px sans-serif`;
        glassCtx.textAlign = 'center';
        glassCtx.textBaseline = 'middle';
        glassCtx.fillText('🐱', logo.x, logo.y - 25 * sf);

        glassCtx.shadowColor = '#ffffff';
        glassCtx.shadowBlur = 10 * sf;
        glassCtx.fillStyle = '#ffffff';
        glassCtx.font = `bold ${Math.round(24 * sf)}px sans-serif`;
        glassCtx.fillText('BeerCat', logo.x, logo.y + 45 * sf);
    }

    glassCtx.restore(); // Exit clipped space
    glassCtx.restore(); // Exit shadow state

    // Draw Crisp Glass elements on top (Loop through all active glasses)
    for (let gIdx = 0; gIdx < glasses.length; gIdx++) {
        const gl = glasses[gIdx];
        const isShot = gl.type === 'shot';

        glassCtx.save();
        glassCtx.translate(gl.x, gl.y);
        glassCtx.rotate(gl.theta);

        // Draw rising fizzy micro bubbles inside the glass (ONLY for pint beer glasses)
        if (!isShot) {
            for (let i = 0; i < activeBubbles; i++) {
                const b = bubblePool[i];
                if (b.glassIndex === gIdx) {
                    glassCtx.fillStyle = `rgba(255, 255, 255, ${b.opacity})`;
                    glassCtx.beginPath();
                    glassCtx.arc(b.u, -b.v, b.radius, 0, Math.PI * 2);
                    glassCtx.fill();
                }
            }
        }

        // Draw Glass outline
        glassCtx.strokeStyle = 'rgba(255, 255, 255, 0.28)';
        glassCtx.lineWidth = isShot ? 3.0 : 5.0;
        glassCtx.beginPath();
        glassCtx.moveTo(-gl.rTop, -gl.height);
        glassCtx.lineTo(-gl.rBase, 0);
        glassCtx.quadraticCurveTo(0, 8, gl.rBase, 0);
        glassCtx.lineTo(gl.rTop, -gl.height);
        glassCtx.stroke();

        // Heavy solid glass base
        glassCtx.fillStyle = 'rgba(255, 255, 255, 0.06)';
        glassCtx.beginPath();
        glassCtx.moveTo(-gl.rBase, 0);
        glassCtx.quadraticCurveTo(0, 5, gl.rBase, 0);
        const baseDepth1 = isShot ? 12 : 22;
        const baseDepth2 = isShot ? 15 : 27;
        glassCtx.lineTo(gl.rBase + 2, baseDepth1);
        glassCtx.quadraticCurveTo(0, baseDepth2, -gl.rBase - 2, baseDepth1);
        glassCtx.closePath();
        glassCtx.fill();

        // Glass specular reflection highlights
        glassCtx.fillStyle = 'rgba(255, 255, 255, 0.12)';
        glassCtx.beginPath();
        const hOffset1 = isShot ? 3 : 8;
        const hOffset2 = isShot ? 6 : 15;
        const hOffset3 = isShot ? 2 : 6;
        const hOffset4 = isShot ? 5 : 13;
        const hOffset5 = isShot ? 6 : 17;
        glassCtx.moveTo(-gl.rTop + hOffset1, -gl.height + hOffset2);
        glassCtx.lineTo(-gl.rBase + hOffset3, -5);
        glassCtx.lineTo(-gl.rBase + hOffset4, -5);
        glassCtx.lineTo(-gl.rTop + hOffset5, -gl.height + hOffset2);
        glassCtx.closePath();
        glassCtx.fill();

        glassCtx.restore();
    }
}

// Pointer event mapping
function getEventPos(e) {
    const rect = container.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
        x: clientX - rect.left,
        y: clientY - rect.top
    };
}

function onStart(e) {
    const pos = getEventPos(e);
    mouse.x = pos.x;
    mouse.y = pos.y;
    mouse.isDown = true;

    let selectedGlass = null;
    let minDistance = Infinity;

    // Find the closest glass within drag bounds
    for (let gIdx = 0; gIdx < glasses.length; gIdx++) {
        const gl = glasses[gIdx];
        const dragYCenter = gl.y - gl.height / 2;
        const dist = Math.sqrt((pos.x - gl.x) * (pos.x - gl.x) + (pos.y - dragYCenter) * (pos.y - dragYCenter));
        const activeRadius = gl.height * 0.9;
        if (dist < activeRadius && dist < minDistance) {
            selectedGlass = gl;
            minDistance = dist;
        }
    }

    if (selectedGlass) {
        selectedGlass.isDragging = true;
        selectedGlass.dragOffsetX = pos.x - selectedGlass.x;
        selectedGlass.dragOffsetY = pos.y - selectedGlass.y;
        
        // Block native window scrolling gestures ONLY if the user clicked inside glass bounds
        if (e.cancelable) {
            e.preventDefault();
        }
    }
}

function onMove(e) {
    const pos = getEventPos(e);
    mouse.x = pos.x;
    mouse.y = pos.y;

    let anyDragging = false;
    for (let gIdx = 0; gIdx < glasses.length; gIdx++) {
        if (glasses[gIdx].isDragging) {
            anyDragging = true;
            break;
        }
    }

    // Block native window scrolling gestures ONLY when dragging a glass
    if (anyDragging && e.cancelable) {
        e.preventDefault();
    }
}

function onEnd() {
    mouse.isDown = false;
    for (let gIdx = 0; gIdx < glasses.length; gIdx++) {
        glasses[gIdx].isDragging = false;
    }
}

container.addEventListener('mousedown', onStart);
window.addEventListener('mousemove', onMove);
window.addEventListener('mouseup', onEnd);

// Registered with { passive: false } to allow dynamic preventDefault scrolling overrides
container.addEventListener('touchstart', onStart, { passive: false });
window.addEventListener('touchmove', onMove, { passive: false });
window.addEventListener('touchend', onEnd);

// Continuous keyboard state listeners
window.addEventListener('keydown', (e) => {
    if (e.key === 'a' || e.key === 'A' || e.key === 'ArrowLeft') keys.Left = true;
    if (e.key === 'd' || e.key === 'D' || e.key === 'ArrowRight') keys.Right = true;
});

window.addEventListener('keyup', (e) => {
    if (e.key === 'a' || e.key === 'A' || e.key === 'ArrowLeft') keys.Left = false;
    if (e.key === 'd' || e.key === 'D' || e.key === 'ArrowRight') keys.Right = false;
});

// Mousewheel continuous tilting (Affects primary glass)
window.addEventListener('wheel', (e) => {
    if (glasses[0]) {
        glasses[0].targetTheta += e.deltaY * 0.0006;
        glasses[0].targetTheta = Math.min(Math.PI * 0.45, Math.max(-Math.PI * 0.45, glasses[0].targetTheta));
    }
});

const tapBtn = document.getElementById('tapBtn');
const resetBtn = document.getElementById('resetBtn');
const spawnShotBtn = document.getElementById('spawnShotBtn');

tapBtn.addEventListener('click', () => {
    isPouring = !isPouring;
    if (isPouring) {
        tapBtn.innerHTML = '<span class="emoji">🛑</span><span class="btn-text">Close Tap</span>';
        tapBtn.style.background = 'rgba(212, 10, 10, 0.25)';
        tapBtn.style.borderColor = 'rgba(212, 10, 10, 0.5)';
    } else {
        tapBtn.innerHTML = '<span class="emoji">🍺</span><span class="btn-text">Open Tap</span>';
        tapBtn.style.background = 'rgba(212, 120, 10, 0.25)';
        tapBtn.style.borderColor = 'rgba(212, 120, 10, 0.5)';
    }
});

resetBtn.addEventListener('click', () => {
    fillGlass();
});

spawnShotBtn.addEventListener('click', () => {
    if (!shotSpawned) {
        // Spawn extra shot glass next to primary pint glass
        const floorY = height - 80;
        glasses.push({
            type: 'shot',
            x: glasses[0].x + 120,
            y: floorY - 15,
            vx: 0, vy: 0, theta: 0, vtheta: 0,
            targetX: glasses[0].x + 120,
            targetY: floorY - 15,
            targetTheta: 0,
            isDragging: false,
            dragOffsetX: 0, dragOffsetY: 0,
            height: 120,
            rBase: 22,
            rTop: 24
        });
        shotSpawned = true;
        spawnShotBtn.innerHTML = '<span class="emoji">❌</span><span class="btn-text">Remove Shot</span>';
        spawnShotBtn.style.background = 'rgba(212, 10, 10, 0.25)';
        spawnShotBtn.style.borderColor = 'rgba(212, 10, 10, 0.5)';
        fillSpecificGlass(1);
    } else {
        // Remove shot glass and all particles/bubbles contained inside it
        if (glasses.length > 1) {
            for (let i = activeCount - 1; i >= 0; i--) {
                if (particlePool[i].insideGlassIndex === 1) {
                    removeParticle(i);
                }
            }
            for (let i = activeBubbles - 1; i >= 0; i--) {
                if (bubblePool[i].glassIndex === 1) {
                    removeBubble(i);
                }
            }
            glasses.splice(1, 1);
        }
        shotSpawned = false;
        spawnShotBtn.innerHTML = '<span class="emoji">🥃</span><span class="btn-text">Spawn Shot</span>';
        spawnShotBtn.style.background = 'rgba(212, 120, 10, 0.25)';
        spawnShotBtn.style.borderColor = 'rgba(212, 120, 10, 0.5)';
    }
});

// Mobile holding tilt buttons
const tiltLeft = document.getElementById('tiltLeft');
const tiltRight = document.getElementById('tiltRight');

tiltLeft.addEventListener('mousedown', () => keys.Left = true);
tiltLeft.addEventListener('mouseup', () => keys.Left = false);
tiltLeft.addEventListener('touchstart', (e) => { e.preventDefault(); keys.Left = true; });
tiltLeft.addEventListener('touchend', () => keys.Left = false);

tiltRight.addEventListener('mousedown', () => keys.Right = true);
tiltRight.addEventListener('mouseup', () => keys.Right = false);
tiltRight.addEventListener('touchstart', (e) => { e.preventDefault(); keys.Right = true; });
tiltRight.addEventListener('touchend', () => keys.Right = false);

function loop() {
    updatePhysics();
    draw();
    requestAnimationFrame(loop);
}

resize();
fillGlass();
loop();
