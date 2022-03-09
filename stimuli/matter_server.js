global.__base = __dirname + '/';

const
  argv = require('minimist')(process.argv.slice(2)),
  app = require('express')(),
  express = require('express'),
  _ = require('lodash');
//   parser = require('xmldom').DOMParser,
//   XMLHttpRequest = require("xmlhttprequest").XMLHttpRequest,
//   sendPostRequest = require('request').post;

var port = process.env.PORT || argv.port || 8885;

app.use(express.static('static'));

server = app.listen(port);
io = require('socket.io')(server);

io.on('connection', function(socket) {
    console.log('connected');
    socket.on('disconnect', function() {
        console.log('disconnected, exiting...');
        process.exit(0);
    });
    socket.on('blocks', function(data) {
        blocks = parseBlocks(data);
        setupWorldWithBlocks(blocks);
        stable = checkStability(data);
        socket.emit('stability', stable);
    });
});


console.log("Created server on port " + port);


var parseBlocks = function(data) {
    // blocks have x, y, w, h
    return blocks;
};

var setupWorldWithBlocks = function(blocks) {
    // clear current world
    World.clear(engine.world);
    // add the floor
    ground = new Block.Boundary(canvasWidth / 2, floorY, canvasWidth * 1.5, floorHeight);
    sideLeft = new Block.Boundary(-30, canvasHeight / 2, 60, canvasHeight);
    sideRight = new Block.Boundary(canvasWidth + 30, canvasHeight / 2, 60, canvasHeight);
    // add each block to the world
    for (var i = 0; i < blocks.length; i++) {
        var block = blocks[i];
        var b = Bodies.rectangle(block.x, block.y, block.w * worldScale, block.h * worldScale, Block.options);
        // NOTE could be done from block kind
        World.add(engine.world, b); //this where a block gets added to matter
    }
};

const MOVEMENT_DELTA = 10;
const ANGLE_DELTA = 0.70;
const SIM_TIME = 3500; //3.5 seconds is what the timeout in the browser is set to
const FRAME_LENGTH = 1000 / 60;

var checkStability = function() {
    // get the starting positions of all blocks
    var start_positions = [];
    for (var i = 0; i < world.bodies.length; i++) {
        var block = world.bodies[i];
        start_positions.push({
            x: block.position.x,
            y: block.position.y,
            angle: block.angle
        });
    }
    // we need to run the engine in increments, long time steps break it
    for (var t = 0; t < SIM_TIME / FRAME_LENGTH; t++) { 
        Engine.update(engine, FRAME_LENGTH); 
        // check if all blocks are in the same place
        for (var i = 3; i < world.bodies.length; i++) { // we can start at 3 here since we ignore the floor and the sides
            var block = world.bodies[i];
            var start_position = start_positions[i];
            // check if block has moved 
            var xMove = Math.abs(block.position.x - start_position.x) > MOVEMENT_DELTA;
            var yMove = Math.abs(block.position.y - start_position.y) > MOVEMENT_DELTA;
            var rotated = Math.abs(block.angle - start_position.angle) > ANGLE_DELTA;
            if (xMove || yMove || rotated) {
                return false;
            }
        }            
    }
    return true;
};

// ---- set up matter world ----

var Matter = require("matter-js");

var Block = require('./block.js');

// Aliases for Matter functions—made global for imported functions
global.Engine = Matter.Engine,
global.World = Matter.World,
global.Bodies = Matter.Bodies,
global.Constraint = Matter.Constraint,
global.MouseConstraint = Matter.MouseConstraint,
global.Mouse = Matter.Mouse,
global.Sleeping = Matter.Sleeping,
global.Runner = Matter.Runner;

// Environment parameters
var canvasHeight = 450;
var canvasWidth = 450;
var menuHeight = canvasHeight / 4.2;
var menuWidth = canvasWidth;
var floorY = (canvasHeight - menuHeight);
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
enableSleeping: true,
velocityIterations: 24,
positionIterations: 12
}

world = World.create({
gravity: {
    y: 2
}
})
global.engine = Engine.create(engineOptions);
var engine = global.engine;
engine.world = world;

// Runner- use instead of line above if changes to game loop needed NOTE probably not needed here
// runner = Matter.Runner.create({
//     isFixed: true
// });
// Runner.run(runner, engine);

console.log("Created engine & world");

// testing code
var test_blocks = [
    {x: 0, y: 590, w: 2, h: 4},
    // {x: 2, y: 670, w: 2, h: 4},
    // {x: 4, y: 4, w: 4, h: 2},
    // {x: 2, y: 7, w: 4, h: 2},
];

setupWorldWithBlocks(test_blocks);

// for (var i = 0; i < 100; i++) {
//     console.log(i,"y:",world.bodies[3].position.y);
//     Engine.update(engine, 1000/60);
// }

stable = checkStability(test_blocks);

console.log("Stability: " + stable);
