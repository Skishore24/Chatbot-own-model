import React from "react";

export default function LeadForm({

  data,

  updateLead,

  submitLeadForm,

  closeLeadForm,

  sendingLead,

  error

}) {

  return (

    <div className="lead-form-card">

      <div className="lead-form-header">

        <div>

          <h3 className="lead-form-title">

            ✉️ Get a Free Quote

          </h3>

          <p className="lead-form-subtitle">

            Leave your details and we'll contact you shortly.

          </p>

        </div>

        <button

          className="lead-close"

          onClick={closeLeadForm}

        >

          <span className="material-symbols-rounded">

            close

          </span>

        </button>

      </div>

      <input

        className="lead-input"

        type="text"

        placeholder="Your Name"

        value={data.name}

        onChange={(e)=>

          updateLead("name", e.target.value)

        }

      />

      <input

        className="lead-input"

        type="email"

        placeholder="Your Email"

        value={data.email}

        onChange={(e)=>

          updateLead("email", e.target.value)

        }

      />

      {

        error && (

          <p className="lead-error">

            {error}

          </p>

        )

      }

      <button

        className="lead-submit"

        onClick={submitLeadForm}

        disabled={sendingLead}

      >

        {

          sendingLead

            ? "Sending..."

            : "Send"

        }

      </button>

    </div>

  );

}