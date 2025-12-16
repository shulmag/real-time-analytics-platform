using FixCommon;
using QuickFix;
using QuickFix.Fields;
using QuickFix.FIX44;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace FixClientLite
{
    public static class MappingHelper
    {     
        public static Trace GetTrace(IndicationOfInterest message)
        {
            Trace o = new Trace();
            #region Description

            if (message.IsSetSymbol())
            {
                o.Symbol = message.Symbol.getValue();
            }

            #endregion

            #region Cusip & BSYM

            if (message.IsSetSecurityID())
            {
                o.Cusip = message.SecurityID.getValue();
            }

            #endregion

            #region OriginalFace

            if (message.IsSetIOIQty())
            {
                o.Quantity = long.Parse(message.IOIQty.getValue());
            }

            #endregion

            #region  Price

            if (message.IsSetPrice())
            {
                var price = message.Price.getValue();
                if (price != -999) o.Price = (double)message.Price.getValue();
            }

            #endregion

            #region Settlement

            if (message.IsSetTransactTime())
            {
                o.SettlementDate = message.TransactTime.getValue();
            }

            #endregion

            #region Other
            // for cancel execution date time is null
            
            if (message.IsSetField(CustomFields.TIMESTAMP_FILED))
            {
                o.ExecutionDateTime =
                DateTime.ParseExact(
                    message.GetString(CustomFields.TIMESTAMP_FILED),
                    CustomFields.DATE_TIME_FORMAT_WITH_MILLISECONDS,
                    CustomFields.DATE_TIME_CULTURE_INFO);
            }
            o.RawFeedId = int.Parse(message.IOIid.getValue());

            if (message.IsSetField(CustomFields.CREATED_DATE_FIELD))
            {
                o.CreatedDate =
                DateTime.ParseExact(
                    message.GetString(CustomFields.CREATED_DATE_FIELD),
                    CustomFields.DATE_TIME_FORMAT_WITH_MILLISECONDS,
                    CustomFields.DATE_TIME_CULTURE_INFO);
            }
            if (message.IsSetField(CustomFields.DISSEMINATIONDATE_FIELD))
            {
                o.DisseminationDate =
                DateTime.ParseExact(
                    message.GetString(CustomFields.DISSEMINATIONDATE_FIELD),
                    CustomFields.DATE_TIME_FORMAT_WITH_MILLISECONDS,
                    CustomFields.DATE_TIME_CULTURE_INFO).Date;
            }
            if (message.IsSetField(CustomFields.FINRA_CREATED_DATE_FIELD))
            {
                o.FinraCreatedDate =
                DateTime.ParseExact(
                    message.GetString(CustomFields.FINRA_CREATED_DATE_FIELD),
                    CustomFields.DATE_TIME_FORMAT_WITH_MILLISECONDS,
                    CustomFields.DATE_TIME_CULTURE_INFO);
            }
            if (message.IsSetField(CustomFields.ORIGINALDISSEMINATIONDATE_FIELD))
            {
                o.OriginalDisseminationDate =
                DateTime.ParseExact(
                    message.GetString(CustomFields.ORIGINALDISSEMINATIONDATE_FIELD),
                    CustomFields.DATE_TIME_FORMAT_WITH_MILLISECONDS,
                    CustomFields.DATE_TIME_CULTURE_INFO).Date;
            }

            Type type = o.GetType();
            foreach (var customField in FinraCustomFields)
            {
                if (message.IsSetField(customField.Key))
                {
                    var val = message.GetString(customField.Key);
                    if (val != CustomFields.NULL_VALUE)
                    {
                        PropertyInfo prop = type.GetProperty(customField.Value);
                        if (prop != null)
                        {
                            var propertyType = prop.PropertyType;

                            if (val == CustomFields.EMPTY_VALUE)
                            {
                                prop.SetValue(o, string.Empty, null);
                            }
                            else if (val == CustomFields.WHITESPACE_VALUE)
                            {
                                prop.SetValue(o, " ", null);
                            }
                            else
                            {
                                var underlyingType = Nullable.GetUnderlyingType(propertyType);
                                if (underlyingType != null)
                                {
                                    propertyType = underlyingType;
                                }

                                if (propertyType.IsEnum)
                                {
                                    var enumValue = Enum.Parse(propertyType, val);
                                    prop.SetValue(o, enumValue);
                                }
                                else
                                {
                                    prop.SetValue(o, Convert.ChangeType(val, propertyType), null);
                                }
                            }
                        }
                    }
                }
            }

            #endregion

            o.RawFixMessageAsXml = message.ToString();
            o.DateInserted = DateTime.UtcNow.ToString();
            return o;
        }

        private static Dictionary<int, string> FinraCustomFields = new Dictionary<int, string>()
        {
            { CustomFields.RDID_FIELD, CustomFields.RDID },
            { CustomFields.QUANTITY_INDICATOR_FIELD, CustomFields.QUANTITY_INDICATOR },
            { CustomFields.REPORTING_PARTY_SIDE_FIELD, CustomFields.REPORTING_PARTY_SIDE },
            { CustomFields.AS_OF_INDICATOR_FIELD, CustomFields.AS_OF_INDICATOR },
           
            { CustomFields.REPORTING_PARTY_TYPE_FIELD, CustomFields.REPORTING_PARTY_TYPE },
            //{ CustomFields.RDID_DESCRIPTION_FIELD, CustomFields.RDID_DESCRIPTION },
           // { CustomFields.RAW_FEED_ID_FIELD, CustomFields.RAW_FEED_ID },
            { CustomFields.BSYM_FIELD, CustomFields.BSYM },
            { CustomFields.UPDATE_DATE_FIELD, CustomFields.UPDATE_DATE },
            { CustomFields.MBS_MESSAGE_TYPE_FIELD, CustomFields.MBS_MESSAGE_TYPE },
            { CustomFields.SUB_PRODUCT_TYPE_FIELD, CustomFields.SUB_PRODUCT_TYPE },
            { CustomFields.BOND_ID_FIELD, CustomFields.BOND_ID },
            { CustomFields.FACTOR_FIELD, CustomFields.FACTOR },
            { CustomFields.SPECIAL_PRICE_INDICATOR_FIELD, CustomFields.SPECIAL_PRICE_INDICATOR },
            { CustomFields.SALE_CONDITION3_FIELD, CustomFields.SALE_CONDITION3 },
            { CustomFields.SALE_CONDITION4_FIELD, CustomFields.SALE_CONDITION4 },
            { CustomFields.CONTRA_PARTY_TYPE_FIELD, CustomFields.CONTRA_PARTY_TYPE },
            { CustomFields.MBSROOT_TRACEID_FIELD, CustomFields.MBSROOT_TRACEID },
            { CustomFields.IS_CANCELED_FIELD, CustomFields.IS_CANCELED },
            { CustomFields.BONDIDENTIFIERS_FIELD, CustomFields.BONDIDENTIFIERS },
            { CustomFields.EXCHANGE_FIELD, CustomFields.EXCHANGE },
            { CustomFields.ORIGINALMESSAGESEQNUMBER_FIELD, CustomFields.ORIGINALMESSAGESEQNUMBER },
           
          
            { CustomFields.HIGHPRICE_FIELD, CustomFields.HIGHPRICE },
            { CustomFields.LOWPRICE_FIELD, CustomFields.LOWPRICE },
            { CustomFields.LASTSALEPRICE_FIELD, CustomFields.LASTSALEPRICE },
            { CustomFields.RENUMERATION_FIELD, CustomFields.RENUMERATION },
            { CustomFields.CHANGEINDICATOR_FIELD, CustomFields.CHANGEINDICATOR },
            { CustomFields.ATSINDICATOR_FIELD, CustomFields.ATSINDICATOR },
           
            { CustomFields.MESSAGESEQNUMBER_FIELD, CustomFields.MESSAGESEQNUMBER }
};
    }

  
}
