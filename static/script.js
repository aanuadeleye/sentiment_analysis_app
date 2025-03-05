document.addEventListener("DOMContentLoaded", function() {
    document.querySelector("form[action='/']").onsubmit = function(event) {
        event.preventDefault();
        let formData = new FormData(this);

        fetch("/", { method: "POST", body: formData })
            .then(response => response.text())
            .then(data => {
                document.getElementById("result").innerHTML = data;
            });
    };
});

