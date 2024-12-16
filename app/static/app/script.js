const currentTab = window.location.href.split("/")[3];
const navLinkElement = $(".nav-link");

if (currentTab === "") {
    $(navLinkElement[0]).addClass("active");
} else {
    navLinkElement.each((index, element) => {
        if (element.text.toLowerCase() === currentTab) {
            $(element).addClass("active");
        }
    });
}

$('input[type="file"]').change(() => {
    $('button[type="submit"]').click();
});

currentService = $('button[type="submit"]').data("id");

if (currentService !== "") {
    document.querySelector("#" + currentService).scrollIntoView();
}