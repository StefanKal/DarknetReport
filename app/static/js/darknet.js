
function complete() {
    currentQuery=$("#myQuery").val()
    $.ajax({
      url: "/search",
      data: {myQuery: currentQuery},
      success: function(results){
        $("#results").html(results);
      }
    });
}

function updateTSPlot(){
	currentQuery=$("massCutoff").val()
    $.ajax({
      url: "/ts_plot",
      data: {currentQuery},
      success: function(results){
	  console.log(results)
	  $("#tsPlotImg").attr("src","data:image/png;base64,"+results);
      }
    });
}

function initTimeSeries() {
	updateListings("Fentanyl HCl");
	updateAvgPrice("Fentanyl HCl");
}

function updateListings(drugname){
    console.log(drugname);
    $.ajax({
      url: "/tsFenListing",
      data: {drugname},
      success: function(results){
      	$("#tsFenListing").attr("src","data:image/png;base64,"+results);
      }
    });
}

function updateAvgPrice(drugname) {
    console.log(drugname);
    $.ajax({
      url: "/tsFenAvgPrice",
      data: {drugname},
      success: function(results){
      	$("#tsFenAvgPrice").attr("src","data:image/png;base64,"+results);
      }
    });
}


$(function(){
    console.log('ready');
    
    $('.list-group li').click(function(e) {
        e.preventDefault()
        var $that = $(this);
        $('.list-group').find('li').removeClass('active');
        $that.addClass('active');
        updateAvgPrice($that.text())
        updateListings($that.text())
    });
})

