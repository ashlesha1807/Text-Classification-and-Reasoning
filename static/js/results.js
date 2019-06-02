
// $(function(){
//     let graphData = {{ data.chart_data | safe }}
//
// });

// let graphData = {{ results | tojson }}
console.log("abcd");
console.log(results);

let bodySelection1 = d3.select("#graph1");

let svgSelection1 = bodySelection1.append("svg")
    .attr("width", 600)
    .attr("height", 600);

let circleSelection1 = svgSelection1.append("circle")
    .attr("cx", 225)
    .attr("cy", 225)
    .attr("r", 125)
    .style("fill", "purple");


let bodySelection2 = d3.select("#graph1");

let svgSelection2 = bodySelection2.append("svg")
    .attr("width", 600)
    .attr("height", 600);

let circleSelection2 = svgSelection2.append("circle")
    .attr("cx", 225)
    .attr("cy", 225)
    .attr("r", 125)
    .style("fill", "purple");



let bodySelection3 = d3.select("#graph1");

let svgSelection3 = bodySelection3.append("svg")
    .attr("width", 600)
    .attr("height", 600);

let circleSelection3 = svgSelection3.append("circle")
    .attr("cx", 225)
    .attr("cy", 225)
    .attr("r", 125)
    .style("fill", "purple");


let bodySelection4 = d3.select("#graph1");

let svgSelection4 = bodySelection4.append("svg")
    .attr("width", 600)
    .attr("height", 600);

let circleSelection4 = svgSelection4.append("circle")
    .attr("cx", 225)
    .attr("cy", 225)
    .attr("r", 125)
    .style("fill", "purple");

