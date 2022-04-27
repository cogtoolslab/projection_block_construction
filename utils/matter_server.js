global.__base = __dirname + "/";

var debug = false;

var busy = false;

var parseBlocks = function (data) {
  // blocks have x, y, w, h
  // for matter, y is upper corner of area. 0,0 is top left, with y decreasing as we go up the area
  // left is x = 0
  // positions specify the midpoint of a rectangle
  var blocks = [];
  for (var i = 0; i < data.length; i++) {
    obj = data[i];
    block = {
      x: x_to_coord(Number(obj.x), Number(obj.w)),
      y: y_to_coord(Number(obj.y), Number(obj.h)),
      w: Number(obj.w),
      h: Number(obj.h),
    };
    blocks.push(block);
  }
  return blocks;
};

var setupWorldWithBlocks = function (blocks) {
  // clear current world
  World.clear(engine.world);
  // add the floor
  ground = new Block.Boundary(
    canvasWidth / 2,
    floorY + DROP_OFFSET,
    canvasWidth * 1.5,
    floorHeight
  );
  // add each block to the world
  for (var i = 0; i < blocks.length; i++) {
    var block = blocks[i];
    var b = Bodies.rectangle(
      block.x,
      block.y,
      block.w * sF * worldScale,
      block.h * sF * worldScale,
      Block.options
    );
    World.add(engine.world, b); //this where a block gets added to matter
  }
};

const MOVEMENT_DELTA = 10;
const ANGLE_DELTA = 0.7;
// const SIM_TIME = 1500; //1.5 seconds is what the timeout in the browser is set to
const SIM_TIME = 5000; //1.5 seconds is what the timeout in the browser is set to
const FRAME_LENGTH = 1000 / 60;
const DROP_OFFSET = 1.72849999999994; // in matter units, how much to drop all blocks to see if they're stable? This value is determined, uh, empirically from the webcode.

var checkStability = function () {
  // get the starting positions of all blocks
  var start_positions = [];
  if (debug) display();
  for (var i = 0; i < world.bodies.length; i++) {
    var block = world.bodies[i];
    start_positions.push({
      x: block.position.x,
      y: block.position.y,
      angle: block.angle,
    });
  }
  // we need to run the engine in increments, long time steps break it
  for (var t = 0; t < SIM_TIME / FRAME_LENGTH; t++) {
    Engine.update(engine, FRAME_LENGTH);
    // check if all blocks are in the same place
    for (var i = 1; i < world.bodies.length; i++) {
      var block = world.bodies[i];
      var start_position = start_positions[i];
      // check if block has moved
      var xMove =
        Math.abs(block.position.x - start_position.x) > MOVEMENT_DELTA;
      var yMove =
        Math.abs(block.position.y - start_position.y) > MOVEMENT_DELTA;
      var rotated = Math.abs(block.angle - start_position.angle) > ANGLE_DELTA;
      if (xMove || yMove || rotated) {
        if (debug) {
          display();
        }
        return false;
      }
    }
  }
  if (debug) {
    display();
  }
  return true;
};

// ---- set up matter world ----

var Matter = require("matter-js");

var Block = require("../utils/block.js");

// Aliases for Matter functions—made global for imported functions
(global.Engine = Matter.Engine),
  (global.World = Matter.World),
  (global.Bodies = Matter.Bodies),
  (global.Constraint = Matter.Constraint),
  (global.MouseConstraint = Matter.MouseConstraint),
  (global.Mouse = Matter.Mouse),
  (global.Sleeping = Matter.Sleeping),
  (global.Runner = Matter.Runner);

// Environment parameters
var canvasHeight = 450;
var canvasWidth = 450;
var menuHeight = canvasHeight / 4.2;
var menuWidth = canvasWidth;
var floorY = canvasHeight - menuHeight;
var floorHeight = canvasHeight / 3;
var aboveGroundProp = floorY / canvasHeight;

// Stimulus parameters
var stimCanvasWidth = canvasWidth;
var stimCanvasHeight = canvasHeight;
var stimX = stimCanvasWidth / 2;
var stimY = stimCanvasHeight / 2;

// Scaling values
var sF = 25; //scaling factor to change appearance of blocks
global.worldScale = 2.2; //scaling factor within matterjs

// Global Variables for Matter js and custom Matter js wrappers
var engine;

// Set up Matter Physics Engine
engineOptions = {
  enableSleeping: false,
  velocityIterations: 24,
  positionIterations: 12,
};

world = World.create({
  gravity: {
    y: 2,
  },
});
global.engine = Engine.create(engineOptions);
var engine = global.engine;
engine.world = world;

var x_to_coord = function (x, w) {
  //okay, so I think the x, y coordinates mark the middle of the rectangle, so we need to take the height into account
  // 137.5 determined empirically (but it also doesn't really matter)
  return 137 + x * sF * worldScale + (w * sF * worldScale) / 2;
};

var y_to_coord = function (y, h) {
  // first term is merely the y position of the floor
  return (floorY - floorHeight / 2 - y * sF - (h * sF) / 2) * worldScale;
};

// DEBUGGING related functions
function getVertices(bodies) {
  // returns the x,y vertices for every body as string
  out_bodies = [];
  for (var i = 0; i < bodies.length; i++) {
    var body = bodies[i];
    var vertices = body.vertices;
    var out_vertices = [];
    for (var j = 0; j < vertices.length; j++) {
      var vertex = vertices[j];
      out_vertices.push([vertex.x, vertex.y]);
    }
    out_bodies.push(out_vertices);
  }
  out_str = JSON.stringify(out_bodies);
  return out_str;
}

var cp = require("child_process");
function display() {
  // calls the python script and blocks till it's closed
  vert_string = getVertices(world.bodies);
  cp.execSync("python utils/matterjs_visualization.py " + vert_string);
}

// read stdin
var stdin = process.openStdin();
stdin.addListener("data", function (d) {
  var msg = d;
  var data = JSON.parse(msg);
  var blocks = parseBlocks(data);
  while (busy) {} // wait for previous simulation to finish
  busy = true;
  setupWorldWithBlocks(blocks);
  var stable = checkStability();
  // write result out to stdout
  console.log(stable);
  busy = false;
});
