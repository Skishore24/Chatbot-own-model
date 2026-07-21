import React from "react";

export default function Suggestions({

  suggestions,

  onSelect

}) {

  if (!suggestions?.length) return null;

  return (

    <div className="suggestions" role="group" aria-label="Quick suggestions">

      {

        suggestions.map((item) => (

          <button

            key={item}

            className="suggestion-btn"

            onClick={() => onSelect(item)}

            aria-label={`Send: ${item}`}

          >

            {item}

          </button>

        ))

      }

    </div>

  );

}