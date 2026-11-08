import { raycast } from './simulation.js';

/** All RGB colors are deliberately broad, stable and texture-free. */
export const PALETTE = Object.freeze({
  ceiling: [30, 45, 46], floor: [52, 58, 53], wall: [188, 187, 168],
  teal: [83, 158, 154], coral: [229, 113, 88], gold: [221, 176, 85],
  team0: [66, 186, 222], team1: [236, 89, 78],
  metal: [125, 140, 135], ink: [35, 48, 48], white: [240, 237, 214],
});

export const SEMANTIC_PALETTE = Object.freeze({
  ceiling: [24, 24, 36], floor: [60, 60, 70], wall: [130, 130, 140],
  ally: [55, 180, 240], enemy: [240, 88, 70],
});

const EYE_HEIGHT = 1;
const WALL_HEIGHT = 2.6;
const MAX_DISTANCE = 64;

/**
 * A CPU renderer shared by browser play and the headless training endpoint.
 * 90° horizontal field of view, RGBA8, nearest-surface visibility, no DOM/GPU.
 * Positive pitch looks upward. Optional labels contain only visible surfaces.
 */
export function renderFrame(sim, { width = 320, height = 180, semantic = false } = {}) {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 64 || width > 1280 || height < 36 || height > 720) {
    throw new RangeError('Frame width must be 64–1280 and height 36–720 integer pixels.');
  }
  if (!sim?.player || !Array.isArray(sim.map)) throw new TypeError('A simulation with player and map is required.');
  const frame = new Uint8ClampedArray(width * height * 4);
  const depth = new Float64Array(width);
  const player = sim.player;
  const f = width * .5;
  const horizon = height * .5 + Math.tan(player.pitch || 0) * f;
  const cos = Math.cos(player.yaw), sin = Math.sin(player.yaw);
  const colors = semantic ? SEMANTIC_PALETTE : PALETTE;

  function put(x, y, rgb, scale = 1) {
    const i = (y * width + x) * 4;
    frame[i] = rgb[0] * scale; frame[i + 1] = rgb[1] * scale; frame[i + 2] = rgb[2] * scale; frame[i + 3] = 255;
  }
  function rectangle(x, y, w, h, rgb) {
    const x0 = Math.max(0, Math.floor(x)), y0 = Math.max(0, Math.floor(y));
    const x1 = Math.min(width, Math.ceil(x + w)), y1 = Math.min(height, Math.ceil(y + h));
    for (let py = y0; py < y1; py++) for (let px = x0; px < x1; px++) put(px, py, rgb);
  }
  function screenPolygon(points, rgb) {
    const minY = Math.max(0, Math.floor(Math.min(...points.map(p => p[1]))));
    const maxY = Math.min(height - 1, Math.ceil(Math.max(...points.map(p => p[1]))));
    for (let y = minY; y <= maxY; y++) {
      const intersections = [];
      for (let k = 0, j = points.length - 1; k < points.length; j = k++) {
        const a = points[j], b = points[k], scanY = y + .5;
        if ((a[1] <= scanY && b[1] > scanY) || (b[1] <= scanY && a[1] > scanY)) {
          intersections.push(a[0] + (scanY - a[1]) * (b[0] - a[0]) / (b[1] - a[1]));
        }
      }
      intersections.sort((a, b) => a - b);
      for (let k = 0; k + 1 < intersections.length; k += 2) {
        for (let x = Math.max(0, Math.ceil(intersections[k])); x < Math.min(width, intersections[k + 1]); x++) put(x, y, rgb);
      }
    }
  }

  // Floor geometry is helpful for estimating motion; its sparse seams fade out
  // with distance, avoiding a high-frequency texture at the vanishing point.
  for (let y = 0; y < height; y++) {
    const floor = y + .5 >= horizon;
    const dist = floor ? f * EYE_HEIGHT / Math.max(.01, y + .5 - horizon) : 0;
    const shade = semantic ? 1 : floor ? .67 + .33 * Math.min(1, 5 / dist) : .85 + .15 * y / height;
    const base = floor ? colors.floor : colors.ceiling;
    for (let x = 0; x < width; x++) {
      let shadeHere = shade;
      if (!semantic && floor && dist < 18) {
        const side = (x + .5 - width / 2) / f;
        const wx = player.x + dist * (cos - side * sin);
        const wy = player.y + dist * (sin + side * cos);
        const gx = Math.abs(wx / 2 - Math.round(wx / 2));
        const gy = Math.abs(wy / 2 - Math.round(wy / 2));
        if (Math.min(gx, gy) < .012) shadeHere *= .8;
      }
      put(x, y, base, shadeHere);
    }
  }

  // Use a perspective plane, not evenly spaced angles, so screen position and
  // hitscan aiming agree over the full field of view.
  for (let x = 0; x < width; x++) {
    const offset = Math.atan((x + .5 - width / 2) / f);
    const hit = raycast(sim, player.x, player.y, player.yaw + offset, MAX_DISTANCE);
    const distance = Math.max(.025, hit.distance * Math.cos(offset));
    depth[x] = distance;
    if (!hit.cell || hit.distance >= MAX_DISTANCE) continue;
    const top = horizon - f * (WALL_HEIGHT - EYE_HEIGHT) / distance;
    const bottom = horizon + f * EYE_HEIGHT / distance;
    const start = Math.max(0, Math.floor(top)), end = Math.min(height, Math.ceil(bottom));
    const accent = hit.cell === '2' ? PALETTE.coral : hit.cell === '3' ? PALETTE.gold : PALETTE.teal;
    const sideShade = hit.side === 1 || hit.side === 'y' ? .83 : 1;
    const shade = sideShade * (.5 + .5 / (1 + distance * .025));
    const coord = hit.side === 1 || hit.side === 'y' ? hit.x : hit.y;
    const seam = Math.abs(coord - Math.round(coord)) < .022;
    for (let y = start; y < end; y++) {
      if (semantic) { put(x, y, SEMANTIC_PALETTE.wall); continue; }
      const fraction = (y + .5 - top) / (bottom - top);
      let material = PALETTE.wall;
      let factor = shade;
      if (fraction > .82) material = PALETTE.ink;
      else if (fraction > .61 && fraction < .75) material = accent;
      else if (fraction < .035) factor *= .77;
      if (seam) factor *= .83;
      put(x, y, material, factor);
    }
  }

  const entities = sim.entities.filter(e => e.active !== false && (e.health ?? 100) > 0 && (e.team === 0 || e.team === 1)).map(e => {
    const dx = e.x - player.x, dy = e.y - player.y;
    return { entity: e, forward: dx * cos + dy * sin, right: -dx * sin + dy * cos };
  }).filter(e => e.forward > .08).sort((a, b) => b.forward - a.forward);

  for (const { entity: e, forward, right } of entities) {
    const worldHeight = e.height || 1.6;
    const worldWidth = 2 * (e.radius || .34);
    const objectWidth = worldWidth * f / forward, objectHeight = worldHeight * f / forward;
    const center = width / 2 + right * f / forward;
    const bottom = horizon + EYE_HEIGHT * f / forward;
    const top = bottom - objectHeight;
    const x0 = Math.max(0, Math.floor(center - objectWidth / 2));
    const x1 = Math.min(width - 1, Math.ceil(center + objectWidth / 2));
    const y0 = Math.max(0, Math.floor(top)), y1 = Math.min(height - 1, Math.ceil(bottom));
    const spriteShade = .72 + .28 / (1 + forward * .015);
    const teamColor = e.team === 0 ? PALETTE.team0 : PALETTE.team1;
    const labelColor = e.team === (player.team ?? 0) ? SEMANTIC_PALETTE.ally : SEMANTIC_PALETTE.enemy;
    const facingViewer = Math.cos((e.yaw || 0) - Math.atan2(player.y - e.y, player.x - e.x)) > 0;
    for (let x = x0; x <= x1; x++) {
      if (forward >= depth[x]) continue;
      const u = (x + .5 - center) / objectWidth;
      const au = Math.abs(u);
      for (let y = y0; y <= y1; y++) {
        const v = (bottom - y - .5) / objectHeight;
        let fill = null;
        // Team color remains visible on helmet, armor, arms and legs. Shape and
        // contrasting chest markings distinguish teams even at small scales.
        const head = au < .205 && v > .82 && v < .99;
        const neck = au < .1 && v > .77 && v <= .84;
        const torso = v > .38 && v <= .81 && au < .25 + .14 * Math.min(1, (v - .38) / .22);
        const arms = v > .45 && v < .73 && au > .3 && au < .49;
        const legs = v > .035 && v <= .4 && au > .055 && au < .295;
        const rifle = u > -.1 && u < .49 && v > .52 && v < .61;
        if (head || neck || torso || arms || legs || rifle) {
          fill = teamColor;
          if (neck || legs && v < .12 || torso && v < .43) fill = PALETTE.ink;
          if (head && facingViewer && v > .875 && v < .925) fill = PALETTE.ink;
          if (torso && au < .25) {
            if (e.team === 0 && v > .67 && v < .72) fill = PALETTE.white;
            if (e.team === 1 && Math.abs(v - (.66 + au * .22)) < .027) fill = PALETTE.white;
          }
          if (rifle) fill = v > .575 ? PALETTE.metal : PALETTE.ink;
          if (e.hitFlash > 0) fill = PALETTE.white;
          else if (e.spawnProtection > 0 && Math.floor(v * 14) % 2 === 0) fill = PALETTE.white;
          put(x, y, semantic ? labelColor : fill, semantic ? 1 : spriteShade);
        }
        if (!semantic && e.muzzleFlash > 0 && u > .36 && u < .5 && Math.abs(v - .565) < .055) {
          put(x, y, PALETTE.gold);
        }
      }
    }
  }

  if (!semantic && player.active !== false && player.health > 0) {
    const s = Math.min(width / 320, height / 180);
    const recoil = player.muzzleFlash > 0 ? 2.5 * s : 0;
    const bx = width * .56, by = height + recoil;
    const sleeve = player.team === 1 ? PALETTE.team1 : PALETTE.team0;
    // Rifle: a long barrel, receiver, top rail, magazine and a colored sleeve.
    // The permanent silhouette stays within the lower fifth of the view.
    const gunPoint = (x, y) => [bx + x * s, by + y * s];
    const gunPolygon = (points, fill) => screenPolygon(points.map(([x,y]) => gunPoint(x,y)), fill);
    if (player.muzzleFlash > 0) {
      gunPolygon([[-3,-30],[-10,-35],[-4,-37],[-7,-44],[1,-39],[7,-44],[7,-37],[14,-35],[6,-31],[2,-24]], PALETTE.gold);
      gunPolygon([[-2,-32],[-3,-38],[3,-37],[6,-34],[2,-29]], PALETTE.white);
    }
    gunPolygon([[6,0],[9,-13],[20,-16],[35,-9],[47,0]], sleeve);
    gunPolygon([[12,0],[8,-11],[18,-16],[24,-8],[27,0]], PALETTE.ink);
    gunPolygon([[-5,-18],[-4,-35],[5,-35],[9,-16]], PALETTE.ink);
    gunPolygon([[-2,-18],[-1,-33],[3,-33],[6,-17]], PALETTE.metal);
    gunPolygon([[-7,-20],[8,-23],[24,-11],[27,0],[-3,0],[-9,-10]], PALETTE.ink);
    gunPolygon([[-5,-19],[6,-21],[19,-10],[17,-3],[-3,-7]], PALETTE.metal);
    gunPolygon([[0,-13],[7,-13],[14,-7],[12,-4],[2,-8]], PALETTE.ink);
    rectangle(bx-2*s,by-26*s,7*s,2*s,PALETTE.ink);
    rectangle(bx,by-28*s,2*s,2*s,sleeve);
    rectangle(bx+5*s,by-18*s,4*s,2*s,PALETTE.gold);
    const cx = Math.floor(width/2), cy = Math.floor(height/2), arm = Math.max(2,Math.round(4*s)), gap = Math.max(1,Math.round(2*s));
    rectangle(cx-gap-arm,cy,arm,1,PALETTE.white); rectangle(cx+gap+1,cy,arm,1,PALETTE.white);
    rectangle(cx,cy-gap-arm,1,arm,PALETTE.white); rectangle(cx,cy+gap+1,1,arm,PALETTE.white);
    if (player.hitMarker > 0) {
      const near = Math.max(3,Math.round(5*s)), far = near + Math.max(2,Math.round(3*s));
      for (let i=near;i<=far;i++) {
        rectangle(cx+i,cy+i,1,1,PALETTE.white); rectangle(cx-i,cy+i,1,1,PALETTE.white);
        rectangle(cx+i,cy-i,1,1,PALETTE.white); rectangle(cx-i,cy-i,1,1,PALETTE.white);
      }
    }
    // Subtle directional-free damage cue; it contains no hidden enemy position.
    if (player.damageFlash > 0) {
      const alpha = Math.min(.28,player.damageFlash*.9), band = Math.max(2,Math.round(5*s));
      for(let y=0;y<height;y++) for(let x=0;x<width;x++) {
        if(x>=band&&x<width-band&&y>=band&&y<height-band) continue;
        const i=(y*width+x)*4;
        frame[i]=frame[i]*(1-alpha)+PALETTE.coral[0]*alpha;
        frame[i+1]=frame[i+1]*(1-alpha)+PALETTE.coral[1]*alpha;
        frame[i+2]=frame[i+2]*(1-alpha)+PALETTE.coral[2]*alpha;
      }
    }
  }
  return frame;
}
